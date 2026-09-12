import pytest
from toolsandbox_pipeline.orchestration.live_environment_attestation import validate_distribution_inventory


def lock():
    return {'package':[dict(name='root',version='1',source={'editable':'.'},dependencies=[{'name':'colorama','marker':"sys_platform == 'win32'"}]),dict(name='colorama',version='2',source={'registry':'https://pypi.org/simple'})]}


def test_platform_dependency_absence_is_explicit():
    result=validate_distribution_inventory(lock(),(('root','1'),),marker_environment={'sys_platform':'linux'})
    assert result['uninstalled_locked_packages']==['colorama']
    assert result['active_locked_dependencies_complete']
    with pytest.raises(ValueError,match='missing active'):
        validate_distribution_inventory(lock(),(('root','1'),),marker_environment={'sys_platform':'win32'})


@pytest.mark.parametrize('installed',[(('root','2'),),(('unknown','1'),),(('root','1'),('root','1'))])
def test_changed_unlocked_duplicate_distribution_is_rejected(installed):
    with pytest.raises(ValueError):
        validate_distribution_inventory(lock(),installed,marker_environment={'sys_platform':'linux'})


def test_all_locked_versions_installed():
    result=validate_distribution_inventory(lock(),(('root','1'),('colorama','2')),marker_environment={'sys_platform':'win32'})
    assert result['uninstalled_locked_packages']==[] and result['version_mismatches']==[]
