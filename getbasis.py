#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 19:25:03 2020

@author: julio

getbasis.py

"""

import numpy as np
import urllib.request
from scipy.special import factorial2
import copy

# =============================================================================
# Objects
# =============================================================================

class BasisFunctions(object):
    ''' A class that contains all our basis function data
        Attributes:
        origin: array/list containing the coordinates of the Gaussian origin
        shell:  orbital angular momentum for the orbital
        exps:   list of primitive Gaussian exponents
        coefs:  list of primitive Gaussian coefficients
        norm:   normalization
    '''
    def __init__(self,origin=[0.0,0.0,0.0],shell=None,exps=None,coefs=None, atom=None, norm=1.):
        self.origin = np.asarray(origin)
        self.shell = shell
        self.exps  = exps
        self.coefs = coefs
        self.norm = norm
        self.atom = atom
        self.oam = None
        self.expn = None
        
    def getNormCo(self):
        return ((2**3)*(self.exps**3)/(np.pi**3))**(1/4)*((4*self.exps)**(self.oam)/factorial2(2*int(self.oam) - 1, exact=True))**(1/2)
        
class ECPbasis(object):
    ''' A class that contains all our basis function data
        Attributes:
        origin: array/list containing the coordinates of the Gaussian origin
        shell:  orbital angular momentum for the orbital
        exps:   list of primitive Gaussian exponents
        coefs:  list of primitive Gaussian coefficients
        norm:   normalization
    '''
    def __init__(self,origin=[0.0,0.0,0.0],r=None,exps=None,coefs=None, atom=None, norm=1., name=None):
        self.origin = np.asarray(origin)
        self.r = r
        self.exps  = exps
        self.coefs = coefs
        self.norm = norm
        self.atom = atom
        self.name = None
        
        
# =============================================================================
# sidekick functions

oam_dict = { 'S':0, 'P':1, 'D':2, 'F':3, 'G':4, 'H':5, 'I':6, 'J':7,    0:'S', 1:'P', 2:'D', 3:'F', 4:'G', 5:'H', 6:'I', 7:'J'   }

oam_dictz = { 'S':0, 'P':1, 'D':2, 'F':3, 'G':4, 'H':5, 'I':6, 'J':7 }

Z_dictonary = np.array([ 'e',  'H', 'He', 
                        'Li', 'Be',  'B',  'C',  'N',  'O',  'F', 'Ne', 
                        'Na', 'Mg', 'Al', 'Si',  'P',  'S', 'Cl', 'Ar',
                         'K', 'Ca', 'Sc', 'Ti',  'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                        'Rb', 'Sr',  'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',  'I', 'Xe', 
                        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',  'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
                        'Fr', 'Ra', 'Ac', 'Th', 'Pa',  'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'])

Element_Names = { 'X':120, 'e':0,   'H':1,  'He':2, 
                 'Li':3,  'Be':4,   'B':5,   'C':6,   'N':7,   'O':8,   'F':9,  'Ne':10, 
                 'Na':11, 'Mg':12, 'Al':13, 'Si':14,  'P':15,  'S':16, 'Cl':17, 'Ar':18,
                  'K':19, 'Ca':20, 'Sc':21, 'Ti':22,  'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36, 
                 'Rb':37, 'Sr':38,  'Y':39, 'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48, 'In':49, 'Sn':50, 'Sb':51, 'Te':52,  'I':53, 'Xe':54, 
                 'Cs':55, 'Ba':56, 'La':57, 'Ce':58, 'Pr':59, 'Nd':60, 'Pm':61, 'Sm':62, 'Eu':63, 'Gd':64, 'Tb':65, 'Dy':66, 'Ho':67, 'Er':68, 'Tm':69, 'Yb':70, 'Lu':71, 'Hf':72, 'Ta':73,  'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80, 'Tl':81, 'Pb':82, 'Bi':83, 'Po':84, 'At':85, 'Rn':86, 
                 'Fr':87, 'Ra':88, 'Ac':89, 'Th':90, 'Pa':91,  'U':92, 'Np':93, 'Pu':94, 'Am':95, 'Cm':96, 'Bk':97, 'Cf':98, 'Es':99, 'Fm':100, 'Md':101, 'No':102, 'Lr':103, 'Rf':104, 'Db':105, 'Sg':106, 'Bh':107, 'Hs':108, 'Mt':109, 'Ds':110, 'Rg':111, 'Cn':112, 'Nh':113, 'Fl':114, 'Mc':115, 'Lv':116, 'Ts':117, 'Og':118}

def Norm(ζ, ℓ):
    return ((2**3)*(ζ**3)/(np.pi**3))**(1/4)*((4*ζ)**(ℓ)/factorial2(2*int(ℓ) - 1, exact=True))**(1/2)

def print_numpy(basis):
    string = ''
    for primitive in basis:
        string += "      " + gbs_format(primitive[0]) + "       " + gbs_format(primitive[1]) + "\n"
        
    return string

def print_ecpnumpy(basis):
    string = ''
    for primitive in basis:
        string += str(int(primitive[0])) + "     " + str(primitive[1]) + "              " + str(primitive[2]) + "\n"
        
    return string

def gbs_format(number):
    # This method will return a number with the proper pading, percision, & notation.
    # This method is written in standard Scienific Notation not Fortan Notation. 
    return np.format_float_scientific(number, unique=False, precision=10, pad_left=2).replace("e", "E") # ()).replace("e", "D")

def prep_name(x):
    
    #urllib.parse.quote_plus(asd)
    x = x.lower()
    x = x.replace("+", "%2B")
    x = x.replace("*", "_st_")
    x = x.replace("(", "%28")
    x = x.replace(")", "%29")
    x = x.replace(" ", "%20")
    x = x.replace("!", "%21")
    x = x.replace("/", "_sl_")
    x = x.replace(",", "%2C")
    
    return x

def write_atom_ecpstring(ecp_basis):

    out  = ""
    out += (ecp_basis[0].atom[0]).split("-")[0] + "     " + str(0) + "\n"
    out += ecp_basis[0].atom[0] + "     " + ecp_basis[0].atom[1] + "     " + ecp_basis[0].atom[2] + "\n"
    for ecp_base in ecp_basis:
        out += ecp_base.name + "\n"
        out += "  " + str(len(ecp_base.r)) + "\n"
        out += print_ecpnumpy(   np.vstack((ecp_base.r, ecp_base.exps, ecp_base.coefs) ).T   )
    
    return out

def get_element_name(name):
    if isinstance(name, str):
        return name
    else:
        return Z_dictonary[name]
    
def write_atom_gbsstring(gbs_basis_list):
    atom_ouput = ""
    #atom_ouput += Z_dictonary[gbs_basis_list[0][0].atom] + "     " + str(0) + "\n"
    atom_ouput += get_element_name(gbs_basis_list[0][0].atom) + "     " + str(0) + "\n"
    for basis in gbs_basis_list:
        atom_ouput +=  oam_dict[basis[0].oam] + "    " + str( len(basis[0].exps) ) + "   " + str(basis[0].norm) + "\n"
        atom_ouput +=  print_numpy( np.vstack( (basis[0].exps, basis[0].coefs) ).T ) #.rstrip('\n')
    atom_ouput += "****\n"
    return atom_ouput

def try_BSElinks(basis_name, version, Z, alt_basis=''):
    """
    This function attempts to download the data from BSE.org
    It trys multiple versions or an alternative basis for a given atom.
    Ultimately it reverts to CRENBL (which is defined for all atoms - He)
    """
    
    try:
        data = urllib.request.urlopen('https://www.basissetexchange.org/api/basis/' + prep_name(basis_name) +'/format/psi4/?version=' + str(version) + '&elements=' + str(Z) )    
    except:
        try:
            data = urllib.request.urlopen('https://www.basissetexchange.org/api/basis/' + prep_name(basis_name) +'/format/psi4/?version=' + str(0) + '&elements=' + str(Z) )    
        except:
            try:
                data = urllib.request.urlopen('https://www.basissetexchange.org/api/basis/' + prep_name(basis_name) +'/format/psi4/?version=' + str(1) + '&elements=' + str(Z) )    
            except:
                try:
                    data = urllib.request.urlopen('https://www.basissetexchange.org/api/basis/' + prep_name(alt_basis) +'/format/psi4/?version=' + str(0) + '&elements=' + str(Z) )    
                except:
                    try:
                        data = urllib.request.urlopen('https://www.basissetexchange.org/api/basis/' + prep_name(alt_basis) +'/format/psi4/?version=' + str(1) + '&elements=' + str(Z) )    
                    except:
                        data = urllib.request.urlopen('https://www.basissetexchange.org/api/basis/' + prep_name("crenbl") +'/format/psi4/?version=' + str(0) + '&elements=' + str(Z) )    
    return data

# =============================================================================
# Get Web (basissetexchange.org) Basis
# =============================================================================

def getBSEBasis(basis_name, Z, version=0, alt_basis=None):
    """
    This function produces a list of Basis Function objects for ECP & GBS types 
        from a BSE URL (given element & basis name).
        
    Input:
        Z:             element number (int)
        basis_name:    name (string)
        version:       basis verison on BSE.org (int) [optional]
    Output:
        
    """
    
    data = try_BSElinks(basis_name, version, Z, alt_basis=alt_basis)

    with data as response:
      html_response = response.read()
      encoding = response.headers.get_content_charset('utf-8')
      decoded_html = html_response.decode(encoding)
      
    decoded_html = decoded_html.split("\n")    
    
    ecp_info     = [] ## other info about basis
    ecp_finder   = [] ## index where found
    basis_finder = [] ## index where found
    basis_info   = [] ## other info about basis
    ecp_basis_finder = []
    
    ### GBS Finder
    whileloop = False
    for i in range(len(decoded_html)):
        if decoded_html[i] == "****":
            whileloop = not whileloop
            continue
                
        if whileloop:
            if len(decoded_html[i].split()) > 0: ## omits white spaces, e.g. \n
                #for oam in line.split()[0]:
                    
                try: ## this checks if the first letter and line are in the dictonary
                    check = oam_dictz[ decoded_html[i].split()[0][0] ] 
    
                    for ind, l in enumerate(decoded_html[i].split()[0]): ## goes through SPD = S, P, D
    
                        number_primatives = int(decoded_html[i].split()[1])
                        primatives_index  = i + np.linspace(1, number_primatives, number_primatives, dtype=int)
                        basis_finder.append( primatives_index )
                        
                        basis_info.append(  [l, number_primatives, float(decoded_html[i].split()[2]), ind]  )
                    
                except:
                    None
    
    ## remove if element name confused basis_finder
    for index, element in enumerate(basis_finder):
        if len(element) == 0:
            pops = index
            
            basis_finder.pop(pops)
            
    ### ECP Finder
    for index, line in enumerate(decoded_html):
        
        if line.find('!') != -1:
            continue
        
        if line.find('ECP') != -1:
            ecp_info.append( line.split() )
             
        if line.find('potential') != -1:
            ecp_info.append( line )
            num = int( decoded_html[index + 1] )
            ecp_primatives_index  = int(index) + np.linspace(1, num, num, dtype=int) + 1
            ecp_finder.append( ecp_primatives_index )
            
    ### get GBS basis
    gbs_basis = []
    for index, element in enumerate(basis_finder):
        
        ## num of basis per OAM
        ℓ = oam_dict[  basis_info[index][0]  ]
        lst = [BasisFunctions() for i in range(2*ℓ+1)]
        
        ## get values
        basis_values =  np.float_(  [ (decoded_html[i].replace("D", "E")).split() for i in element]  )
        
        ## initialize orbtials
        for newbasis in lst:
            newbasis.atom  = Z
            newbasis.exps  = basis_values[:, 0]
            newbasis.coefs = basis_values[:, basis_info[index][3] + 1 ] ## this needs to be 2 sometimes...
            newbasis.oam   = ℓ
            newbasis.norm  = basis_info[index][2]
            
        gbs_basis.append(lst)
    
    ### get ECP basis
    ecp_basis = []
    for index, element in enumerate(ecp_finder):
        basis_valuez =  np.float_(  [ (decoded_html[i].replace("D", "E")).split() for i in element]  )
        new_ecp = ECPbasis()
        
        new_ecp.r     = basis_valuez.T[0]
        new_ecp.exps  = basis_valuez.T[1]
        new_ecp.coefs = basis_valuez.T[2]
        new_ecp.name  = ecp_info[1 + index]
        new_ecp.atom  = ecp_info[0]
        
        ecp_basis.append(new_ecp)
    
    return gbs_basis, ecp_basis

# =============================================================================
# Get GBS Basis From File
# =============================================================================

def get_file_basis(name, path=''):
    
    file = open( path + name + '.gbs',"r")
    lines = file.readlines()
    file.close()
    
    ### this trims the gbs file
    for ind, line in enumerate(lines):
        if line == '****\n':
            break
    lines = lines[ind+1:]
    
    atom_basis = []
    atom_info = []
    basis_finder = []
    basis_info = []
    
    Z_name = lines[0].split()[0]
    for i in range(len(lines)):
        if lines[i] == '****\n' or lines[i] == "****":
            
            atom_basis.append(basis_finder)
            atom_info.append(basis_info)
            
            basis_finder = []
            basis_info = []
            
            try:
                Z_name = lines[i+1].split()[0]
            
            except:
                continue
            
            continue
    
        if len(lines[i].split()) > 1: ## omits white spaces, e.g. \n
            #for oam in line.split()[0]:
            if lines[i].split()[1] == "0":
                continue
                
            try: ## this checks if the first letter and line are in the dictonary
                check = oam_dictz[ lines[i].split()[0][0] ] 
    
                for ind, l in enumerate(lines[i].split()[0]): ## goes through SPD = S, P, D
    
                    number_primatives = int(lines[i].split()[1])
                    primatives_index  = i + np.linspace(1, number_primatives, number_primatives, dtype=int)
                    
                    basis_finder.append( primatives_index )
                    basis_info.append(  [l, number_primatives, float(lines[i].split()[2]), ind, Z_name]  )
                
            except:
                None

    atom_base_obj = []
    for i_atom, atom in enumerate(atom_basis):
        
        gbs_basis_list = []
        for index, element in enumerate(atom):
            
            ## num of basis per OAM
            ℓ = oam_dict[  atom_info[i_atom][index][0]  ]
            lst = [BasisFunctions() for i in range(2*ℓ+1)]
            
            ## get values
            basis_values =  np.float_(  [ (lines[i].replace("D", "E")).split() for i in element]  )
            
            ## initialize orbtials
            for newbasis in lst:
                newbasis.atom  = atom_info[i_atom][index][4] #!!!
                newbasis.exps  = basis_values[:, 0]
                newbasis.coefs = basis_values[:, atom_info[i_atom][index][3] + 1 ] ## this needs to be 2 sometimes...
                newbasis.oam   = ℓ
                newbasis.norm  = atom_info[i_atom][index][2]
            
            gbs_basis_list.append(lst)
    
        atom_base_obj.append(gbs_basis_list)

    element_name = np.empty(len(atom_base_obj), dtype="<U2" )
    for i in range(len(atom_base_obj)):
        element_name[i] = atom_base_obj[i][0][0].atom

    return atom_base_obj, element_name

# =============================================================================
# !!! Main Program !!!
# =============================================================================

def get_GBSfile(input_basis, Zset, alt_basis=None, file=None, file_dir='', empty=None, version=0, output_gbs="newbasis"):
    Zset = np.unique(Zset)
    
    ### reexamine Zset for empty
    if empty is not None:
        for i, ele in enumerate(Zset):
            if ele == empty:
                Zset.pop(i)
    
    ### get gbs, ecp objects for a set of atoms
    atom_gbs = []
    atom_ecp = []
    for element in Zset:
        gbs_temp, ecp_temp = getBSEBasis(input_basis, element, version, alt_basis=alt_basis)
        atom_gbs.append(gbs_temp)
        atom_ecp.append(ecp_temp)
        
    ### Get from File, if file is None:
    if file is not None:
        gbs_base, element_name_list = get_file_basis(file, path=file_dir) ## !!! only does gbs
        for index, element in enumerate(element_name_list):
            if len(np.where(Z_dictonary[Zset] == element)[0]) > 0:
                location_BSE = np.where(Z_dictonary[Zset] == element)[0][0]
                temp = gbs_base[0] + atom_gbs[location_BSE]
                atom_gbs[location_BSE] = temp
                
            else:
                atom_gbs.append(gbs_base[index])
                atom_ecp.append([])
        
    ### print out gbs objects to .gbs file
    output = "****\n"
    for i, atom in enumerate(atom_gbs):
        output += write_atom_gbsstring(atom)
    
    ### print empty atom
    if empty is not None:
        output += Z_dictonary[empty] + "     " + str(0) + "\n" + "****"
    
    ### print ecp to gbs
    out_ecp = ""
    for i, atom in enumerate(atom_ecp): ## run thru all atoms! some might be empty!!
        if atom: ## check if it is empty
            out_ecp += write_atom_ecpstring(atom) + "\n\n"
    
    output += "\n\n" + out_ecp
    
    ## write file with 'output' string
    file = open(output_gbs + '.gbs',"w")
    file.write( output )
    file.close()
    
    return atom_gbs, atom_ecp ## atom gbs: ATOM, SHELL, 2n+1 basis functions

# =============================================================================
# XYZ Geometry
# =============================================================================

def getBasis_xyz(Z_ex, xyz, atom_gbs, X):
    
    ## get Znum selection from function (as a directory for the available basis functions)
    Znum = []
    for atom in atom_gbs:
        if isinstance(atom[0][0].atom, np.int64) or isinstance(atom[0][0].atom, int):
            Znum.append( atom[0][0].atom )
        else:
            Znum.append( Element_Names.get( atom[0][0].atom , -1 ) )
    
    ## get basis set with coordinates (atom geometry) !!! error?
    atoms_geometry = []
    count = 0 #count number of basis functions
    for atom_index, atom in enumerate(Z_ex):
        temp_atom = copy.deepcopy( atom_gbs[  np.where(atom == Znum)[0][0]  ] )
        for shell in temp_atom:
            for basisfun in shell:
                basisfun.origin = xyz[atom_index]
                count += 1
        atoms_geometry.append(temp_atom)
    
    ## make cube grid, on which to project the density/φ_μxyz
    oX = np.einsum('i, j, k -> ijk', X, np.ones(len(X), dtype=int), np.ones(len(X), dtype=int))
    oY = np.einsum('i, j, k -> ijk', np.ones(len(X), dtype=int), X, np.ones(len(X), dtype=int))
    oZ = np.einsum('i, j, k -> ijk', np.ones(len(X), dtype=int), np.ones(len(X), dtype=int), X)
    
    ## calculate φ_μxyz (basis function, x-coorinate, y-coorinate, z-coorinate)
    φ_μxyz = np.zeros( (count, oX.shape[0], oY.shape[1], oZ.shape[2] ) )
    i = 0
    for atom_index, atom in enumerate(atoms_geometry):
        R2 = (oX - atom[0][0].origin[0])**2 + (oY - atom[0][0].origin[1])**2 + (oZ - atom[0][0].origin[2])**2
        for shell_index, shell in enumerate(atom):
            for base_index, base in enumerate(shell):
                φ_μxyz[i] = np.einsum('a, a, xyz, axyz -> xyz', base.coefs,  Norm(base.exps, (base.oam) ), R2**(base.oam/2),   np.exp( - np.einsum('a, xyz -> axyz', base.exps, R2 ) )   )/np.sqrt(2*(base.oam) + 1) 
                i += 1
    return φ_μxyz

# =============================================================================
# Add with Geometry
# =============================================================================

def BasisGeo(Znum, gbs_list, xyz):
    atomgbs_atoms = np.zeros(len(gbs_list), dtype=int)
    for index, atom in enumerate(gbs_list):
        atomgbs_atoms[index] = int( atom[0][0].atom )
    
    geometry_list = []
    for atom in Znum:
        geometry_list.append( gbs_list[np.where(atomgbs_atoms == atom)[0][0] ] )
        
    ### set xyz coordinates on basis (give geometry)
    atoms_geometry = []
    for atom_index, atom in enumerate(geometry_list):
        temp_atom = copy.deepcopy( gbs_list[  np.where(atomgbs_atoms == atom[0][0].atom )[0][0]  ] )
        for shell in temp_atom:
            for basisfun in shell:
                basisfun.origin = xyz[atom_index]
        atoms_geometry.append(temp_atom) ## this is a lists-of-lists-of-lists
    
    ## flatten atoms_geometry
    flat_list = []
    for sublist in atoms_geometry:
        for item in sublist:
            for subitem in item:
                flat_list.append(subitem)
    
    return flat_list

def flatten_atomgbs(atoms_geometry):
    flat_list = []
    for sublist in atoms_geometry:
        for item in sublist:
            for subitem in item:
                flat_list.append(subitem)
    return flat_list

# =============================================================================
# For Testing/Sanity Checks
# =============================================================================

atom_gbs, atom_ecp = get_GBSfile("LANL2DZ", [1,8])
