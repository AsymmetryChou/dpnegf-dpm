# Hamiltonian
import json

import numpy as np
import torch
import pytest
from dptb.nn.build import build_model

from dpnegf.negf.negf_hamiltonian_init import NEGFHamiltonianInit
from dpnegf.utils.make_kpoints import kmesh_sampling
from dpnegf.negf.device_property import DeviceProperty
from dpnegf.negf.lead_property import LeadProperty

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)


def test_negf_Hamiltonian(root_directory):

    model_ckpt=root_directory +'/dpnegf/tests/data/test_negf/test_negf_run/nnsk_C.json'
    results_path=root_directory +"/dpnegf/tests/data/test_negf/test_negf_hamiltonian/test_negf_hamiltonian_init"
    input_path = root_directory +"/dpnegf/tests/data/test_negf/test_negf_hamiltonian/run_input.json"
    structure=root_directory +"/dpnegf/tests/data/test_negf/test_negf_run/chain.vasp"
    # log_path=root_directory +"/dptb/tests/data/test_negf/test_negf_Device/test.log"
    negf_json = json.load(open(input_path))
    model = build_model(model_ckpt,model_options=negf_json['model_options'],common_options=negf_json['common_options'])


    hamiltonian = NEGFHamiltonianInit(model=model,
                                    AtomicData_options=negf_json['AtomicData_options'], 
                                    structure=structure,
                                    pbc_negf = negf_json['task_options']["stru_options"]['pbc'], 
                                    stru_options = negf_json['task_options']['stru_options'],
                                    unit = negf_json['task_options']['unit'], 
                                    results_path=results_path,
                                    torch_device = torch.device('cpu'),
                                    block_tridiagonal=negf_json['task_options']['block_tridiagonal'])

    # hamiltonian = NEGFHamiltonianInit(apiH=apiHrk, structase=structase, stru_options=task_options["stru_options"], results_path=results_path)
    kpoints= kmesh_sampling(negf_json['task_options']["stru_options"]["kmesh"])
    with torch.no_grad():
        struct_device, struct_leads, _,_,_ = hamiltonian.initialize(kpoints=kpoints)

    
    deviceprop = DeviceProperty(hamiltonian, struct_device, results_path=results_path, efermi=negf_json['task_options']['e_fermi'])
    deviceprop.set_leadLR(
            lead_L=LeadProperty(
            hamiltonian=hamiltonian, 
            tab="lead_L", 
            structure=struct_leads["lead_L"], 
            results_path=results_path,
            e_T=negf_json['task_options']['ele_T'],
            efermi=negf_json['task_options']['e_fermi'], 
            voltage=negf_json['task_options']["stru_options"]["lead_L"]["voltage"]
        ),
            lead_R=LeadProperty(
            hamiltonian=hamiltonian, 
            tab="lead_R", 
            structure=struct_leads["lead_R"], 
            results_path=results_path,
            e_T=negf_json['task_options']['ele_T'],
            efermi=negf_json['task_options']['e_fermi'], 
            voltage=negf_json['task_options']["stru_options"]["lead_R"]["voltage"]
        )
    )

    leads = ["lead_L", "lead_R"]
    e=0  
    for ll in leads:
        getattr(deviceprop, ll).self_energy(
            energy=e, 
            kpoint=kpoints[0], 
            eta_lead=negf_json['task_options']["eta_lead"],
            method=negf_json['task_options']["sgf_solver"]
            )
    print("lead_L self energy:",deviceprop.lead_L.se)
    print("lead_R self energy:",deviceprop.lead_R.se)

    lead_L_se_standard = torch.tensor([[1.8103e-08-0.6096j, 0.0000e+00+0.0000j, 0.0000e+00+0.0000j,
         0.0000e+00+0.0000j],
         [0.0000e+00+0.0000j, 0.0000e+00+0.0000j, 0.0000e+00+0.0000j,
         0.0000e+00+0.0000j],
        [0.0000e+00+0.0000j, 0.0000e+00+0.0000j, 0.0000e+00+0.0000j,
         0.0000e+00+0.0000j],
        [0.0000e+00+0.0000j, 0.0000e+00+0.0000j, 0.0000e+00+0.0000j,
         0.0000e+00+0.0000j]], dtype=torch.complex128)
    assert abs(deviceprop.lead_L.se-lead_L_se_standard).max()<1e-5

    lead_R_se_standard = torch.tensor([[0.0000e+00+0.0000j, 0.0000e+00+0.0000j, 0.0000e+00+0.0000j,
         0.0000e+00+0.0000j],
        [0.0000e+00+0.0000j, 0.0000e+00+0.0000j, 0.0000e+00+0.0000j,
         0.0000e+00+0.0000j],
        [0.0000e+00+0.0000j, 0.0000e+00+0.0000j, 0.0000e+00+0.0000j,
         0.0000e+00+0.0000j],
        [0.0000e+00+0.0000j, 0.0000e+00+0.0000j, 0.0000e+00+0.0000j,
        1.8103e-08-0.6096j]], dtype=torch.complex128)
    assert abs(deviceprop.lead_R.se-lead_R_se_standard).max()<1e-5

    #check device's Hamiltonian
    assert all(struct_device.symbols=="C4")
    assert all(struct_device.pbc)==False
    assert np.diag(np.array(struct_device.cell==[10.0, 10.0, 19.2])).all()
    
    #check lead_L's Hamiltonian
 
    assert all(struct_leads["lead_L"].symbols=="C4")
    assert struct_leads["lead_L"].pbc[0]==False
    assert struct_leads["lead_L"].pbc[1]==False
    assert struct_leads["lead_L"].pbc[2]==True
    assert np.diag(np.array(struct_leads["lead_L"].cell==[10.0, 10.0, -6.4])).all()
    
    #check lead_R's Hamiltonian
 
    assert all(struct_leads["lead_R"].symbols=="C4")
    assert struct_leads["lead_R"].pbc[0]==False
    assert struct_leads["lead_R"].pbc[1]==False
    assert struct_leads["lead_R"].pbc[2]==True
    assert np.diag(np.array(struct_leads["lead_R"].cell==[10.0, 10.0, 6.4])).all()


    #check hs_device
    h_device = hamiltonian.get_hs_device(kpoint=np.array([0,0,0]),V=0,block_tridiagonal=False)[0][0]
    print(hamiltonian.get_hs_device(kpoint=np.array([0,0,0]),V=0,block_tridiagonal=False)[0])
    h_device_standard = torch.tensor([[-13.6386+0.j,   0.6096+0.j,   0.0000+0.j,   0.0000+0.j],
        [  0.6096+0.j, -13.6386+0.j,   0.6096+0.j,   0.0000+0.j],
        [  0.0000+0.j,   0.6096+0.j, -13.6386+0.j,   0.6096+0.j],
        [  0.0000+0.j,   0.0000+0.j,   0.6096+0.j, -13.6386+0.j]],dtype=torch.complex128)
    assert abs(h_device-h_device_standard).max()<1e-4

    s_device = hamiltonian.get_hs_device(kpoint=np.array([0,0,0]),V=0,block_tridiagonal=False)[1][0]
    print(hamiltonian.get_hs_device(kpoint=np.array([0,0,0]),V=0,block_tridiagonal=False)[1][0])
    s_standard = torch.eye(4)
    assert abs(s_device-s_standard).max()<1e-5



    #check hs_lead
    hl_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[0][0]
    hl_lead_standard = torch.tensor([-13.6386+0.j,   0.6096+0.j], dtype=torch.complex128)    
    assert abs(hl_lead-hl_lead_standard).max()<1e-4

    hll_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[1][0]
    hll_lead_standard = torch.tensor([0.0000+0.j, 0.6096+0.j], dtype=torch.complex128)
    print(hll_lead)    
    assert abs(hll_lead-hll_lead_standard).max()<1e-4

    hDL_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[2]
    hDL_lead_standard = torch.tensor([[0.0000+0.j, 0.6096+0.j],
        [0.0000+0.j, 0.0000+0.j],
        [0.0000+0.j, 0.0000+0.j],
        [0.0000+0.j, 0.0000+0.j]], dtype=torch.complex128)   
    assert abs(hDL_lead-hDL_lead_standard).max()<1e-5

    sl_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[3]
    sl_lead_standard = torch.eye(2)   
    assert abs(sl_lead-sl_lead_standard).max()<1e-5

    sll_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[4]
    sll_lead_standard = torch.zeros(2)    
    assert abs(sll_lead-sll_lead_standard).max()<1e-5

    sDL_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[5]
    sDL_lead_standard = torch.zeros([4,2])    
    assert abs(sDL_lead-sDL_lead_standard).max()<1e-5   


    # check device norbs
    na = len(hamiltonian.device_norbs)
    device_norbs_standard=[1,1,1,1]
    assert na == 4
    assert hamiltonian.device_norbs==device_norbs_standard

def test_calc_principal_layers_disp_vec():
    '''
    unittest for static method calc_principal_layers_disp_vec of class
    NEGFHamiltonianInit
    '''
    symm_thr = 1e-5
    # test the following cases
    # case 1: normal case
    coords = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]])
    out = NEGFHamiltonianInit.calc_principal_layers_disp_vec(coords, symm_thr)
    assert np.allclose(out, np.array([0, 0, 1]), atol=1e-6)
    
    # case 2: with odd number of atoms
    coords_nat_odd = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 1]])
    try:
        NEGFHamiltonianInit.calc_principal_layers_disp_vec(coords_nat_odd, symm_thr)
    except ValueError as e:
        assert 'The number of atoms in the lead structure must be even for' in str(e)
    
    # case 3: with atoms have not consistent displacement vector
    coords_error = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 2]])
    try:
        NEGFHamiltonianInit.calc_principal_layers_disp_vec(coords_error, symm_thr)
    except ValueError as e:
        assert 'principal layers of one lead to be translationally equivalent' in str(e)

    # case 4: with atoms have not consistent displacement vector but with a small error
    coords_equiv = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1 + 1e-6]])
    out = NEGFHamiltonianInit.calc_principal_layers_disp_vec(coords_equiv, symm_thr)
    assert np.allclose(out, np.array([0, 0, 1]), atol=1e-6)
    # case 4-1: however, if the error is larger than symm_thr, it will raise an error
    try:
        NEGFHamiltonianInit.calc_principal_layers_disp_vec(coords_equiv, 1e-7)
    except ValueError as e:
        assert 'principal layers of one lead to be translationally equivalent' in str(e)

# def _test_negf_Hamiltonian(root_directory):

#     model_ckpt=root_directory +'/dptb/tests/data/test_negf/test_negf_run/nnsk_C.json'
#     jdata = root_directory +"/dptb/tests/data/test_negf/test_negf_hamiltonian/run_input.json"
#     structure=root_directory +"/dptb/tests/data/test_negf/test_negf_run/chain.vasp"
#     log_path=root_directory +"/dptb/tests/data/test_negf/test_negf_hamiltonian/test.log"

#     apihost = NNSKHost(checkpoint=model_ckpt, config=jdata)
#     apihost.register_plugin(InitSKModel())
#     apihost.build()
#     apiHrk = NN2HRK(apihost=apihost, mode='nnsk')
#     jdata = j_loader(jdata)
#     task_options = j_must_have(jdata, "task_options")

#     run_opt = {
#             "run_sk": True,
#             "init_model":model_ckpt,
#             "results_path":root_directory +"/dptb/tests/data/test_negf/test_negf_hamiltonian/",
#             "structure":structure,
#             "log_path": log_path,
#             "log_level": 5,
#             "use_correction":False
#         }


#     structase=read(run_opt['structure'])
#     results_path=run_opt.get('results_path')
#     kpoints=np.array([[0,0,0]])

#     hamiltonian = NEGFHamiltonianInit(apiH=apiHrk, structase=structase, stru_options=task_options["stru_options"], results_path=results_path)
#     with torch.no_grad():
#         struct_device, struct_leads = hamiltonian.initialize(kpoints=kpoints)
    
#     #check device's Hamiltonian
#     assert all(struct_device.symbols=="C4")
#     assert all(struct_device.pbc)==False
#     assert np.diag(np.array(struct_device.cell==[10.0, 10.0, 19.2])).all()
    
#     #check lead_L's Hamiltonian
 
#     assert all(struct_leads["lead_L"].symbols=="C4")
#     assert struct_leads["lead_L"].pbc[0]==False
#     assert struct_leads["lead_L"].pbc[1]==False
#     assert struct_leads["lead_L"].pbc[2]==True
#     assert np.diag(np.array(struct_leads["lead_L"].cell==[10.0, 10.0, -6.4])).all()
    
#     #check lead_R's Hamiltonian
 
#     assert all(struct_leads["lead_R"].symbols=="C4")
#     assert struct_leads["lead_R"].pbc[0]==False
#     assert struct_leads["lead_R"].pbc[1]==False
#     assert struct_leads["lead_R"].pbc[2]==True
#     assert np.diag(np.array(struct_leads["lead_R"].cell==[10.0, 10.0, 6.4])).all()


#     #check hs_device
#     h_device = hamiltonian.get_hs_device(kpoint=np.array([0,0,0]),V=0,block_tridiagonal=False)[0][0]
#     print(hamiltonian.get_hs_device(kpoint=np.array([0,0,0]),V=0,block_tridiagonal=False)[0])
#     h_device_standard = torch.tensor([[-13.6386+0.j,   0.6096+0.j,   0.0000+0.j,   0.0000+0.j],
#         [  0.6096+0.j, -13.6386+0.j,   0.6096+0.j,   0.0000+0.j],
#         [  0.0000+0.j,   0.6096+0.j, -13.6386+0.j,   0.6096+0.j],
#         [  0.0000+0.j,   0.0000+0.j,   0.6096+0.j, -13.6386+0.j]],dtype=torch.complex128)
#     assert abs(h_device-h_device_standard).max()<1e-4

#     s_device = hamiltonian.get_hs_device(kpoint=np.array([0,0,0]),V=0,block_tridiagonal=False)[1][0]
#     print(hamiltonian.get_hs_device(kpoint=np.array([0,0,0]),V=0,block_tridiagonal=False)[1][0])
#     s_standard = torch.eye(4)
#     assert abs(s_device-s_standard).max()<1e-5



#     #check hs_lead
#     hl_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[0][0]
#     hl_lead_standard = torch.tensor([-13.6386+0.j,   0.6096+0.j], dtype=torch.complex128)    
#     assert abs(hl_lead-hl_lead_standard).max()<1e-4

#     hll_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[1][0]
#     hll_lead_standard = torch.tensor([0.0000+0.j, 0.6096+0.j], dtype=torch.complex128)
#     print(hll_lead)    
#     assert abs(hll_lead-hll_lead_standard).max()<1e-4

#     hDL_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[2]
#     hDL_lead_standard = torch.tensor([[0.0000+0.j, 0.6096+0.j],
#         [0.0000+0.j, 0.0000+0.j],
#         [0.0000+0.j, 0.0000+0.j],
#         [0.0000+0.j, 0.0000+0.j]], dtype=torch.complex128)   
#     assert abs(hDL_lead-hDL_lead_standard).max()<1e-5

#     sl_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[3]
#     sl_lead_standard = torch.eye(2)   
#     assert abs(sl_lead-sl_lead_standard).max()<1e-5

#     sll_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[4]
#     sll_lead_standard = torch.zeros(2)    
#     assert abs(sll_lead-sll_lead_standard).max()<1e-5

#     sDL_lead = hamiltonian.get_hs_lead(kpoint=np.array([0,0,0]),tab="lead_L",v=0)[5]
#     sDL_lead_standard = torch.zeros([4,2])    
#     assert abs(sDL_lead-sDL_lead_standard).max()<1e-5   


#     # check device norbs
#     na = len(hamiltonian.device_norbs)
#     device_norbs_standard=[1,1,1,1]
#     assert na == 4
#     assert hamiltonian.device_norbs==device_norbs_standard

def test_sort_device_lexico_preserves_species():
    """Regression test for the device-sort species desync bug.

    The block-tridiagonal device sort must permute the whole atoms slice so
    that atomic species stay attached to their coordinates. A previous version
    permuted only ``positions``, which for a non-pre-sorted, multi-species
    device relocated species onto the wrong sites and corrupted the
    Hamiltonian.
    """
    from ase import Atoms
    from dpnegf.negf.negf_hamiltonian_init import NEGFHamiltonianInit
    from dpnegf.negf.sort_btd import sort_lexico

    # leads: 2 C atoms each; device: mixed C/H that is NOT lexicographically
    # ordered along z, so a correct sort must actually permute it.
    #   idx: 0  1 | 2   3   4   5 | 6  7
    #   sym: C  C | C   H   C   H | C  C
    #   z  : 0  1 | 5   3   4   2 | 8  9
    symbols = ["C", "C", "C", "H", "C", "H", "C", "C"]
    zs = [0.0, 1.0, 5.0, 3.0, 4.0, 2.0, 8.0, 9.0]
    positions = [[0.0, 0.0, z] for z in zs]
    atoms = Atoms(symbols=symbols, positions=positions,
                  cell=[10.0, 10.0, 10.0], pbc=[True, True, False])
    device_id = [2, 6]

    # ground-truth mapping computed independently
    dev_pos = np.array(positions)[device_id[0]:device_id[1]]
    perm = sort_lexico(dev_pos)
    expected_dev_sym = np.array(symbols[device_id[0]:device_id[1]])[perm].tolist()
    expected_dev_z = dev_pos[perm][:, 2].tolist()

    sorted_atoms = NEGFHamiltonianInit._sort_device_lexico(atoms, device_id)
    got_sym = list(sorted_atoms.get_chemical_symbols())
    got_z = sorted_atoms.positions[:, 2].tolist()

    # device region: species stay attached to their (now sorted) coordinates
    assert got_sym[device_id[0]:device_id[1]] == expected_dev_sym
    assert np.allclose(got_z[device_id[0]:device_id[1]], expected_dev_z)
    # device coordinates are actually sorted ascending in z
    assert got_z[device_id[0]:device_id[1]] == sorted(got_z[device_id[0]:device_id[1]])
    # H atoms remain H at the correct sorted coordinates (z=2 and z=3)
    dev_pairs = list(zip(got_sym[device_id[0]:device_id[1]],
                         got_z[device_id[0]:device_id[1]]))
    assert ("H", 2.0) in dev_pairs and ("H", 3.0) in dev_pairs

    # leads are untouched (verbatim)
    assert got_sym[:device_id[0]] == symbols[:device_id[0]]
    assert got_sym[device_id[1]:] == symbols[device_id[1]:]
    assert np.allclose(got_z[:device_id[0]], zs[:device_id[0]])
    assert np.allclose(got_z[device_id[1]:], zs[device_id[1]:])

    # cell and pbc preserved
    assert np.allclose(sorted_atoms.cell, atoms.cell)
    assert list(sorted_atoms.pbc) == list(atoms.pbc)   

