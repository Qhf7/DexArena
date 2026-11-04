"""Compact inference overlay CPU regressions; no JAX/GPU needed."""
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from train.train_configs.pi05_config import DEXJOCO_COMMIT, TrainingContract
from utils.pi05_utils import compact


def seal(root, config):
    config['trainable_sha256'] = compact.sha(root/'ckpt/trainable.npz')
    (root/'config.json').write_text(json.dumps(config))
    records = [dict(path=p,bytes=(root/p).stat().st_size,sha256=compact.sha(root/p)) for p in ('config.json','ckpt/trainable.npz')]
    (root/'manifest.json').write_text(json.dumps(dict(status='EXACT_ALL_PARAMETER_BYTES_RECONSTRUCTION_PASS',identity_sha256=config['identity_sha256'],base_manifest_sha256=config['base_manifest_sha256'],files=records)))


@pytest.fixture
def package(tmp_path, monkeypatch):
    root=tmp_path/'package';(root/'ckpt').mkdir(parents=True)
    base=tmp_path/'base';(base/'params').mkdir(parents=True);(base/'params/data').write_bytes(b'pinned-base')
    values={'backbone/frozen':np.array([1,2],dtype=np.float16),'expert/projection':np.array([3,4],dtype=np.float32),'expert/lora':np.array([5,6],dtype=np.float32)}
    np.savez(root/'ckpt/trainable.npz',**{k:v for k,v in values.items() if not k.endswith('frozen')})
    monkeypatch.setattr(compact,'restore_base',lambda _: {'backbone':{'frozen':values['backbone/frozen'].astype(np.float32)}})
    names=[f'joint{i}' for i in range(29)]
    config=dict(schema_version=1,method='pi05',format='pi05_trainable_npz_v1',inference_only=True,optimizer_state_included=False,checkpoint='ckpt/trainable.npz',source_commits={'dexjoco':DEXJOCO_COMMIT},training=dataclasses.asdict(TrainingContract()),optimizer_updates=30000,identity_sha256='identity',base_manifest_sha256='base-identity',base_manifest=[dict(name='checkpoints/pi05_base/params/data',bytes=11,sha256=compact.sha(base/'params/data'))],parameter_manifest=[dict(path=k,shape=list(v.shape),dtype=str(v.dtype),trainable=not k.endswith('frozen'),sha256=compact.array_sha(v)) for k,v in values.items()],task='bolt',env_id='NutBoltPickBolt-Sharpa-v1',data_contract=dict(task='bolt',env_id='NutBoltPickBolt-Sharpa-v1',state_names=names,action_names=names,controller_names=names,stats={kind:{stat:[1]*29 for stat in ('mean','std','q01','q99')} for kind in ('state','actions')}))
    seal(root,config)
    expected={'backbone':{'frozen':np.zeros(2,np.float16)},'expert':{'projection':np.zeros(2,np.float32),'lora':np.zeros(2,np.float32)}}
    return root,base,config,expected,values


def test_exact_overlay_includes_non_lora_and_preserves_dtype(package):
    root,base,_,expected,values=package
    loaded=compact.flatten(compact.load_compact_params(root,base,expected))
    for path,value in values.items():
        assert loaded[path].dtype==value.dtype
        assert loaded[path].tobytes()==value.tobytes()


def test_model_entry_binds_exact_tree_without_optimizer(package, monkeypatch):
    from algo import pi05

    root,base,_,expected,values=package
    seen={}
    model=SimpleNamespace(eval=lambda:seen.update(evaluated=True))
    def bind(params,remove_extra_params):
        assert remove_extra_params is False
        seen['params']=compact.flatten(params)
        return model
    monkeypatch.setattr(pi05,'expected_model_tree',lambda dual:(SimpleNamespace(load=bind),expected))
    monkeypatch.setitem(sys.modules,'jax',SimpleNamespace(device_put=lambda params,device:params,block_until_ready=lambda params:None))
    actual,prepared,certificate=pi05.load_compact_model(root,base,device='cpu')
    assert actual is model and seen['evaluated']
    assert prepared['env_id']=='NutBoltPickBolt-Sharpa-v1'
    assert certificate['inference_only'] is True
    assert set(seen['params'])==set(values)


@pytest.mark.parametrize('change',['missing','extra','wrong_shape','wrong_dtype','nonfinite','wrong_bytes'])
def test_bad_trainable_archive_rejected(package,change):
    root,base,cfg,expected,values=package
    arrays={k:v.copy() for k,v in values.items() if not k.endswith('frozen')}
    if change=='missing':del arrays['expert/projection']
    elif change=='extra':arrays['unexpected']=np.zeros(1)
    elif change=='wrong_shape':arrays['expert/projection']=np.zeros(3,np.float32)
    elif change=='wrong_dtype':arrays['expert/projection']=arrays['expert/projection'].astype(np.float64)
    elif change=='nonfinite':arrays['expert/projection'][0]=np.nan
    else:arrays['expert/projection'][0]=999
    np.savez(root/'ckpt/trainable.npz',**arrays);seal(root,cfg)
    with pytest.raises(ValueError):compact.load_compact_params(root,base,expected)


@pytest.mark.parametrize('change',['duplicate','missing','extra','shape','dtype','flag'])
def test_bad_parameter_manifest_rejected(package,change):
    root,base,cfg,expected,_=package
    rows=cfg['parameter_manifest']
    if change=='duplicate':rows.append(dict(rows[0]))
    elif change=='missing':rows.pop()
    elif change=='extra':rows.append(dict(rows[0],path='unexpected'))
    elif change=='shape':rows[0]['shape']=[3]
    elif change=='dtype':rows[0]['dtype']='float64'
    else:rows[0]['trainable']=1
    seal(root,cfg)
    with pytest.raises(ValueError):compact.load_compact_params(root,base,expected)


@pytest.mark.parametrize('change',['changed','extra','escape','duplicate'])
def test_base_admission_rejected(package,change):
    root,base,cfg,expected,_=package
    if change=='changed':(base/'params/data').write_bytes(b'other-base!')
    elif change=='extra':(base/'params/extra').write_bytes(b'extra')
    elif change=='escape':cfg['base_manifest'][0]['name']='../outside'
    else:cfg['base_manifest'].append(dict(cfg['base_manifest'][0]))
    seal(root,cfg)
    with pytest.raises(ValueError):compact.load_compact_params(root,base,expected)


def test_package_corruption_rejected(package):
    root,_,_,_,_=package
    with (root/'ckpt/trainable.npz').open('ab') as stream:stream.write(b'bad')
    with pytest.raises(ValueError,match='checksum'):compact.read_compact_config(root)


def test_certificate_path_traversal_rejected(package):
    root,_,_,_,_=package
    value=json.loads((root/'manifest.json').read_text());value['files'][0]['path']='../outside'
    (root/'manifest.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):compact.read_compact_config(root)


@pytest.mark.parametrize('change',['source','format','update','normalization','identity','resume'])
def test_contract_mismatch_rejected(package,change):
    root,_,cfg,_,_=package
    if change=='source':cfg['source_commits']['dexjoco']='wrong'
    elif change=='format':cfg['format']='torch'
    elif change=='update':cfg['optimizer_updates']=29999
    elif change=='normalization':cfg['data_contract']['stats']['state']['mean']=[1]
    elif change=='identity':cfg['data_contract']['env_id']='other'
    else:cfg['optimizer_state_included']=True
    seal(root,cfg)
    with pytest.raises(ValueError):compact.read_compact_config(root)
