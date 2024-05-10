from numpy import pi, linspace
tet_ang = 0.30408672#radians


class scsd_symmetry:
    def __init__(self, ptgr):
        self.ptgr = ptgr
        self.ops  = point_group_dict.get(self.ptgr)
        self.pgt  = pgt_dict.get(self.ptgr)
        self.ops_order  = ordered_ops_dict.get(self.ptgr)
        self.ops_verb = [operations_dict.get(x) for x in self.ops]
        self.ops_verb_order = [operations_dict.get(x) for x in self.ops_order]
        self.e_group_parent = e_group_parent.get(self.ptgr)
        self.e_group_parent_multi = e_group_parent_multi.get(self.ptgr)
        self.e_group_unique_sets = e_group_unique_sets.get(self.ptgr)
        self.mondrian_lookup_dict = mondrian_lookup_dict.get(self.ptgr)
        self.mondrian_orientation_dict = mondrian_orientation_dict.get(self.ptgr)
        self.mondrian_transform_dict = mondrian_transform_dict.get(self.ptgr, self.mondrian_lookup_dict)
        self.irrep_typog = irrep_typog.get(self.ptgr)
        self.irrep_typog_html = irrep_typog_html.get(self.ptgr)
        self.symm_multiplicity_tables = symm_multiplicity_tables.get(self.ptgr)


operations_dict = {'ident' :[['rotation',0,0,1]],
                   'C2'    :[['rotation',0,0,2]],
                   'C3'    :[['rotation',0,0,3]],
                   'C4'    :[['rotation',0,0,4]],
                   'C5'    :[['rotation',0,0,5]],
                   'C6'    :[['rotation',0,0,6]],
                   'C7'    :[['rotation',0,0,7]],
                   'C8'    :[['rotation',0,0,8]],
                   '2C3'   :[['rotation',0,0,3],['rotation',0,pi,3]],
                   '2C4'   :[['rotation',0,0,4],['rotation',0,pi,4]],
                   '2C5'   :[['rotation',0,0,5],['rotation',0,pi,5]],
                   '2C6'   :[['rotation',0,0,6],['rotation',0,pi,6]],
                   '2C8'   :[['rotation',0,0,8],['rotation',0,pi,8]],
                   'C3^2'  :[['rotation',0,0,3/2]],
                   'C4^3'  :[['rotation',0,0,4/3]],
                   'C5^2'  :[['rotation',0,0,5/2]],
                   'C5^3'  :[['rotation',0,0,5/3]],
                   'C5^4'  :[['rotation',0,0,5/4]],
                   'C6^5'  :[['rotation',0,0,6/5]],
                   'C7^2'  :[['rotation',0,0,7/2]],
                   'C7^3'  :[['rotation',0,0,7/3]],
                   'C7^4'  :[['rotation',0,0,7/4]],
                   'C7^5'  :[['rotation',0,0,7/5]],
                   'C7^6'  :[['rotation',0,0,7/6]],
                   'C8^3'  :[['rotation',0,0,8/3]],
                   'C8^5'  :[['rotation',0,0,8/5]],
                   'C8^7'  :[['rotation',0,0,8/7]],
                   '2C5^2' :[['rotation',0,0,5/2],['rotation',0,pi,5/2]],
                   '2C7^2' :[['rotation',0,0,7/2],['rotation',0,pi,7/2]],
                   '2C7^3' :[['rotation',0,0,7/3],['rotation',0,pi,7/3]],
                   '2C8^3' :[['rotation',0,0,8/3],['rotation',0,pi,8/3]],
                   'C2*'   :[['rotation',0,pi/2,2]],
                   '2C2*'  :[['rotation',x,pi/2, 2] for x in linspace(0,pi*(1/2),2)],
                   '2C2*1' :[['rotation',0,pi/2, 2]],
                   '2C2*2' :[['rotation',pi/2,pi/2, 2]],
                   '2C2*3' :[['rotation',pi/3,pi/2, 2]],
                   '2C2*4' :[['rotation',pi*2/3,pi/2, 2]],
                   '3C2*'  :[['rotation',x,pi/2, 2] for x in linspace(0,pi*(2/3),3)],
                   '4C2*'  :[['rotation',x,pi/2, 2] for x in linspace(0,pi*(3/4),4)],
                   '5C2*'  :[['rotation',x,pi/2, 2] for x in linspace(0,pi*(4/5),5)],
                   '6C2*'  :[['rotation',x,pi/2, 2] for x in linspace(0,pi*(5/6),6)],
                   '7C2*'  :[['rotation',x,pi/2, 2] for x in linspace(0,pi*(6/7),7)],
                   '8C2*'  :[['rotation',x,pi/2, 2] for x in linspace(0,pi*(7/8),8)],
                   
                   '2C2**' :[['rotation',pi/4,pi/2,2],['rotation',pi*3/4,pi/2,2]],
                   '2C2**1' :[['rotation',pi/4,pi/2,2]],
                   '2C2**2' :[['rotation',pi*3/4,pi/2,2]],
                   '3C2**' :[['rotation',pi/6,pi/2,2],['rotation',pi/2,pi/2,2],['rotation',pi*5/6,pi/2,2]],
                   '4C2**' :[['rotation',pi/8,pi/2,2],['rotation',pi*3/8,pi/2,2],['rotation',pi*5/8,pi/2,2],['rotation',pi*7/8,pi/2,2]],
                   'C2**'  :[['rotation',pi/2,pi/2,2]],
                   #for d8h
                   '4C2*1'  :[['rotation',0,pi/2, 2]],
                   '4C2*2'  :[['rotation',pi/4,pi/2, 2]],
                   '4C2*3'  :[['rotation',pi/2,pi/2, 2]],
                   '4C2*4'  :[['rotation',pi*3/4,pi/2, 2]],
                   '4C2**1' :[['rotation',pi/8,pi/2,2]],
                   '4C2**2' :[['rotation',pi*3/8,pi/2,2]],
                   '4C2**3' :[['rotation',pi*5/8,pi/2,2]],
                   '4C2**4' :[['rotation',pi*7/8,pi/2,2]],
                   '4sig_v1'   :[['mirror',pi/2,pi/2, 2]],
                   '4sig_v2'   :[['mirror',pi*3/4,pi/2, 2]],
                   '4sig_v3'   :[['mirror',pi,pi/2, 2]],
                   '4sig_v4'   :[['mirror',pi*5/4,pi/2, 2]],
                   
                   'i'     :[['inversion',0,0,1]],
                   
                   'S3'    :[['improperrotation',0,0,3]],
                   'S3^5'  :[['improperrotation',0,0,3/5]],
                   'S4'    :[['improperrotation',0,0,4]],
                   'S4^3'  :[['improperrotation',0,0,4/3]],
                   'S5'    :[['improperrotation',0,0,5]],
                   'S5^3'  :[['improperrotation',0,0,5/3]],
                   'S5^7'  :[['improperrotation',0,0,5/7]],
                   'S5^9'  :[['improperrotation',0,0,5/9]],
                   'S6'    :[['improperrotation',0,0,6]],
                   'S6^5'  :[['improperrotation',0,0,6/5]],
                   'S8'    :[['improperrotation',0,0,8]],
                   'S8^3'  :[['improperrotation',0,0,8/3]],
                   'S8^5'  :[['improperrotation',0,0,8/5]],
                   'S8^7'  :[['improperrotation',0,0,8/7]],
                   'S10'   :[['improperrotation',0,0,10]],
                   'S10^3' :[['improperrotation',0,0,10/3]],
                   'S10^7' :[['improperrotation',0,0,10/7]],
                   'S10^9' :[['improperrotation',0,0,10/9]],
                   'S12'   :[['improperrotation',0,0,12]],
                   'S12^5' :[['improperrotation',0,0,12/5]],
                   'S12^7' :[['improperrotation',0,0,12/7]],
                   'S12^11':[['improperrotation',0,0,12/11]],
                   
                   '2S3'   :[['improperrotation',0,0,3],['improperrotation',0,pi,3]],
                   '2S4'   :[['improperrotation',0,0,4],['improperrotation',0,pi,4]],
                   '2S5'   :[['improperrotation',0,0,5],['improperrotation',0,pi,5]],
                   '2S5^3' :[['improperrotation',0,0,5/3],['improperrotation',0,pi,5/3]],
                   '2S6'   :[['improperrotation',0,0,6],['improperrotation',0,pi,6]],
                   '2S7'   :[['improperrotation',0,0,7],['improperrotation',0,pi,7]],
                   '2S7^3' :[['improperrotation',0,0,7/3],['improperrotation',0,pi,7/3]],
                   '2S7^5' :[['improperrotation',0,0,7/5],['improperrotation',0,pi,7/5]],
                   '2S8'   :[['improperrotation',0,0,8],['improperrotation',0,pi,8]],
                   '2S8^3' :[['improperrotation',0,0,8/3],['improperrotation',0,pi,8/3]],
                   '2S12'  :[['improperrotation',0,0,12],['improperrotation',0,pi,12]],
                   '2S12^5':[['improperrotation',0,0,12/5],['improperrotation',0,pi,12/5]],
                   '2S14'  :[['improperrotation',0,0,14],['improperrotation',0,pi,14]],
                   '2S14^3':[['improperrotation',0,0,14/3],['improperrotation',0,pi,14/3]],
                   '2S14^5':[['improperrotation',0,0,14/5],['improperrotation',0,pi,14/5]],
                   '2S16'  :[['improperrotation',0,0,16],['improperrotation',0,pi,16]],
                   '2S16^3':[['improperrotation',0,0,16/3],['improperrotation',0,pi,16/3]],
                   '2S16^5':[['improperrotation',0,0,16/5],['improperrotation',0,pi,16/5]],
                   '2S16^7':[['improperrotation',0,0,16/7],['improperrotation',0,pi,16/7]],
                   
                   'sig_h'    :[['mirror',0,0,2]],
                   '2sig_v'   :[['mirror',pi/2,pi/2, 2],['mirror',pi,pi/2, 2]],
                   '2sig_v1'  :[['mirror',pi/2,pi/2, 2]],
                   '2sig_v2'  :[['mirror',pi,pi/2, 2]],
                   '3sig_v'   :[['mirror',x,pi/2, 2] for x in linspace(pi/2,pi*(1/2+2/3),3)],
                   '3sig_v1'  :[['mirror',pi/2,pi/2, 2]],
                   '3sig_v2'  :[['mirror',pi*(5/6),pi/2, 2]],
                   '3sig_v3'  :[['mirror',pi*(7/6),pi/2, 2]],
                   '4sig_v'   :[['mirror',x,pi/2, 2] for x in linspace(pi/2,pi*(1/2+3/4),4)],
                   '5sig_v'   :[['mirror',x,pi/2, 2] for x in linspace(pi/2,pi*(1/2+4/5),5)],
                   '7sig_v'   :[['mirror',x,pi/2, 2] for x in linspace(pi/2,pi*(1/2+6/7),7)],
                     
                   '2sig_d'   :[['mirror',x,pi/2, 2] for x in linspace(pi*1/4,pi*3/4,2)],       
                   '2sig_d1'  :[['mirror',pi*1/4,pi/2, 2]],       
                   '2sig_d2'  :[['mirror',pi*3/4,pi/2, 2]],       
                   '2sig_d3'  :[['mirror',0,pi/2, 2]],       
                   '2sig_d4'  :[['mirror',pi*1/3,pi/2, 2]],       
                   '2sig_d5'  :[['mirror',pi*2/3,pi/2, 2]], 
                   '4sig_d1'  :[['mirror',pi*1/8,pi/2, 2]], 
                   '4sig_d2'  :[['mirror',pi*3/8,pi/2, 2]], 
                   '4sig_d3'  :[['mirror',pi*5/8,pi/2, 2]], 
                   '4sig_d4'  :[['mirror',pi*7/8,pi/2, 2]], 
                   '3sig_d'   :[['mirror',x,pi/2, 2] for x in linspace(0,pi*2/3,3)],          
                   '4sig_d'   :[['mirror',x,pi/2, 2] for x in linspace(pi*1/8,pi*7/8,4)],          
                   '5sig_d'   :[['mirror',x,pi/2, 2] for x in linspace(0,pi*4/5,5)],
                   '6sig_d'   :[['mirror',x,pi/2, 2] for x in linspace(pi*1/12,pi*11/12,6)],
                   '7sig_d'   :[['mirror',x,pi/2, 2] for x in linspace(0,pi*6/7,7)],
                   '8sig_d'   :[['mirror',x,pi/2, 2] for x in linspace(pi*1/16,pi*15/16,8)],
                   'sig_v(xz)':[['mirror', pi/2,pi/2,2]],
                   'sig_v(yz)':[['mirror', 0,pi/2,2]],
                   
                   #unfinished = special groups
                   '8C31'  :[['rotation',pi*1/4,pi*tet_ang, 3]],
                   '8C32'  :[['rotation',pi*3/4,pi*tet_ang, 3]],
                   '8C33'  :[['rotation',pi*5/4,pi*tet_ang, 3]],
                   '8C34'  :[['rotation',pi*7/4,pi*tet_ang, 3]],
                   '8C35'  :[['rotation',pi*1/4,pi*(1-tet_ang), 3]],
                   '8C36'  :[['rotation',pi*3/4,pi*(1-tet_ang), 3]],
                   '8C37'  :[['rotation',pi*5/4,pi*(1-tet_ang), 3]],
                   '8C38'  :[['rotation',pi*7/4,pi*(1-tet_ang), 3]],
                   
                   '6S41'  :[['improperrotation',0,0, 4]],
                   '6S42'  :[['improperrotation',0,pi/2, 4]],
                   '6S43'  :[['improperrotation',pi/2,pi/2, 4]],
                   '6S44'  :[['improperrotation',pi,pi/2, 4]],
                   '6S45'  :[['improperrotation',pi*3/2,pi/2, 4]],
                   '6S46'  :[['improperrotation',0,pi, 4]],
                   
                   '3C21'  :[['rotation',0,0, 2]],
                   '3C22'  :[['rotation',0,pi/2, 2]],
                   '3C23'  :[['rotation',pi/2,pi/2, 2]],

                   '6sigd1'  :[['mirror',pi*1/4,pi/2, 2]], 
                   '6sigd2'  :[['mirror',pi*3/4,pi/2, 2]],
                   '6sigd3'  :[['mirror',pi,pi/4, 2]],
                   '6sigd4'  :[['mirror',pi,pi*3/4, 2]],
                   '6sigd5'  :[['mirror',pi/2,pi/4, 2]],
                   '6sigd6'  :[['mirror',pi/2,pi*3/4, 2]],

                   '6C21'  :[['rotation',0,     pi/4,   2]],
                   '6C22'  :[['rotation',pi/2,  pi/4,   2]],
                   '6C23'  :[['rotation',pi,    pi/4,   2]],
                   '6C24'  :[['rotation',pi*3/2,pi/4,   2]],
                   '6C25'  :[['rotation',pi/4,  pi/2,   2]],
                   '6C26'  :[['rotation',pi*3/4,pi/2,   2]],

                   '6C41'  :[['rotation',0,     0,   4]],
                   '6C42'  :[['rotation',0,     pi,  4]],
                   '6C43'  :[['rotation',0,     pi/2,4]],
                   '6C44'  :[['rotation',pi,    pi/2,4]],
                   '6C45'  :[['rotation',pi/2,  pi/2,4]],
                   '6C46'  :[['rotation',pi*3/2,pi/2,4]],

                   '8S61'  :[['improperrotation',pi*1/4,pi*tet_ang, 6]],
                   '8S62'  :[['improperrotation',pi*3/4,pi*tet_ang, 6]],
                   '8S63'  :[['improperrotation',pi*5/4,pi*tet_ang, 6]],
                   '8S64'  :[['improperrotation',pi*7/4,pi*tet_ang, 6]],
                   '8S65'  :[['improperrotation',pi*1/4,pi*(1-tet_ang), 6]],
                   '8S66'  :[['improperrotation',pi*3/4,pi*(1-tet_ang), 6]],
                   '8S67'  :[['improperrotation',pi*5/4,pi*(1-tet_ang), 6]],
                   '8S68'  :[['improperrotation',pi*7/4,pi*(1-tet_ang), 6]],

                   '3C2*1'  :[['rotation',0,pi/2, 2]],
                   '3C2*2'  :[['rotation',pi*1/3,pi/2, 2]],
                   '3C2*3'  :[['rotation',pi*2/3,pi/2, 2]],
                   '3C2**1'  :[['rotation',pi*3/6,pi/2, 2]],
                   '3C2**2'  :[['rotation',pi*5/6,pi/2, 2]],
                   '3C2**3'  :[['rotation',pi*1/6,pi/2, 2]],
                   '3sig_d1':[['mirror',0,pi/2, 2]],       
                   '3sig_d2':[['mirror',pi*1/3,pi/2, 2]],       
                   '3sig_d3':[['mirror',pi*2/3,pi/2, 2]],  

                   '5C2*1'  : [['rotation',0,pi/2, 2]],
                   '5C2*2'  : [['rotation',pi*1/5,pi/2, 2]],
                   '5C2*3'  : [['rotation',pi*2/5,pi/2, 2]],
                   '5C2*4'  : [['rotation',pi*3/5,pi/2, 2]],
                   '5C2*5'  : [['rotation',pi*4/5,pi/2, 2]],
                   '5sig_v1': [['mirror',pi*1/2,pi/2, 2]],
                   '5sig_v2': [['mirror',pi*7/10,pi/2, 2]],
                   '5sig_v3': [['mirror',pi*9/10,pi/2, 2]],
                   '5sig_v4': [['mirror',pi*1/10,pi/2, 2]],
                   '5sig_v5': [['mirror',pi*3/10,pi/2, 2]],
                  
  
                     '2C7'      : [['rotation', 0, 0, 7],  ['rotation', 0, pi, 7]],
                     '2C72'     : [['rotation', 0, 0, 7/2],['rotation', 0, pi, 7/2]],
                     '2C73'     : [['rotation', 0, 0, 7/3],['rotation', 0, pi, 7/3]],
                     '7C2*1'    : [['rotation', 0, pi/2, 2]],   
                     '7C2*2'    : [['rotation', pi*1/7, pi/2, 2]],  
                     '7C2*3'    : [['rotation', pi*2/7, pi/2, 2]],  
                     '7C2*4'    : [['rotation', pi*3/7, pi/2, 2]],  
                     '7C2*5'    : [['rotation', pi*4/7, pi/2, 2]],  
                     '7C2*6'    : [['rotation', pi*5/7, pi/2, 2]],  
                     '7C2*7'    : [['rotation', pi*6/7, pi/2, 2]],
                     '2S71'     : [['improperrotation', 0, 0, 7],  ['improperrotation', 0, pi, 7] ],
                     '2S73'     : [['improperrotation', 0, 0, 7/3],['improperrotation', 0, pi, 7/3]],
                     '2S75'     : [['improperrotation', 0, 0, 7/5],['improperrotation', 0, pi, 7/5]],
                     '7sig_v1'  : [['mirror', pi*7/14, pi/2, 2]],   
                     '7sig_v2'  : [['mirror', pi*9/14, pi/2, 2]],   
                     '7sig_v3'  : [['mirror', pi*11/14, pi/2, 2]],  
                     '7sig_v4'  : [['mirror', pi*13/14, pi/2, 2]],  
                     '7sig_v5'  : [['mirror', pi*15/14, pi/2, 2]],  
                     '7sig_v6'  : [['mirror', pi*17/14, pi/2, 2]],  
                     '7sig_v7'  : [['mirror', pi*19/14, pi/2, 2]],

                   '12C5'  :[['rotation',0,0,5]] +
                            [['rotation',x,pi*0.35283333, 5] for x in linspace(pi/10,9*pi/10,5)],
                   '15C2'  :[['rotation',x,pi/2, 2] for x in linspace(0,4*pi/5,5)] +
                            [['rotation',x,pi*0.17641666, 2] for x in linspace(pi/10,9*pi/10,5)] +
                            [['rotation',x,pi*(1-0.17641666), 2] for x in linspace(pi/10,9*pi/10,5)],
                  }

subgroups_dict   = {'C1' :  [],
                    'Cs' :  'C1'.split(','),
                    'Ci' :  'C1'.split(','),
                    'C2' :  'C1'.split(','),
                    'C3' :  'C1'.split(','),
                    'C4' :  'C2,C1'.split(','),
                    'C5' :  'C1'.split(','),
                    'C6' :  'C1,C2,C3'.split(','),
                    'C7' :  'C1'.split(','),
                    'C8' :  'C1,C2,C4'.split(','),
                    'D2' :  'C1,C2'.split(','),
                    'D3' :  'C1,C2,C3'.split(','),
                    'D4' :  'C1,C2,C4,D2'.split(','),
                    'D5' :  'C1,C2,C5'.split(','),
                    'D6' :  'C1,C2,C3,C6,D2,D3'.split(','),
                    'D7' :  'C1,C2,C7'.split(','),
                    'D8' :  'C1,C2,C4,C8,D2,D4'.split(','),
                    'C2v':  'C1,C2,Cs'.split(','),
                    'C3v':  'C3,Cs,C1'.split(','),
                    'C4v':  'C4,C2v,C2,Cs,C1'.split(','),
                    'C5v':  'C1,Cs,C5'.split(','),
                    'C6v':  'C6,C3v,C3,C2v,C2,Cs,C1'.split(','),
                    'C7v':  'C1,Cs,C7'.split(','),
                    'C8v':  'Cs,C2,C4,C8,C2v,C4v,C1'.split(','),
                    'C2h':  'C2,Ci,Cs,C1'.split(','),
                    'C3h':  'C3,Cs,C1'.split(','),
                    'C4h':  'C4,S4,C2h,C2,Cs,Ci,C1'.split(','),
                    'C5h':  'C1,Cs,C5'.split(','),
                    'C6h':  'S6,C3h,C6,C3,C2h,Cs,C2,Ci,C1'.split(','),
                    'D2h':  'C2v,D2,C2h'.split(','),
                    'D3h':  'C3h,C3v,D3,C3,C2v,C2,Cs,C1'.split(','),
                    'D4h':  'D2d,C4v,D4,C4h,C4,S4,D2h,C2v,D2,C2h,Cs,C2,Ci,C1'.split(','),
                    'D5h':  'Cs,C2,C5,D5,C2v,C5v,C5h,C1'.split(','),
                    'D6h':  'S6,D3h,C6v,D6,C6h,D3d,C3h,C6,C3v,D3,C3,D2h,C2v,D2,C2h,C2,Cs,Ci,C1'.split(','),
                    'D7h':  'Cs,C2,C7,D7,C2v,C7v,C7h,C1'.split(','),
                    'D8h':  'C1,Cs,Ci,C2,C4,C8,D2,D4,D8,C2v,C4v,C8v,C2h,C4h,C8h,D2h,D4h,D2d,D4d,S4,S8'.split(','),
                    'D2d':  'S4,C2v,D2,C2,Cs,C1'.split(','),   
                    'D3d':  'S6,C3v,D3,C3,C2h,Cs,C2,Ci,C1'.split(','),
                    'D4d':  'Cs,C2,C4,D2,D4,C2v,C4v,S8,C1'.split(','),
                    'D5d':  'C1,Cs,Ci,C2,C5,D5,C5v,C2h,S10'.split(','),
                    'D6d':  'C1,Cs,C2,C3,C6,D2,D3,D6,C2v,C3v,C6v,D2d,S4,S12'.split(','),
                    'D7d':  'C1,Cs,Ci,C2,C7,D7,C2v,C2h,S14'.split(','),
                    'D8d':  'C1,Cs,C2,C4,C8,D2,D4,D8,C2v,C4v,C8v,S16'.split(','),
                    'S2' :  'C1'.split(','),
                    'S4' :  'C1,C2'.split(','),
                    'S6' :  'C1,Ci,C3'.split(','),#aka C3i
                    'S8' :  'C1,C2,C4'.split(','),
                    'S10':  'C1,Ci,C5'.split(','), 
                    'S12':  'C1,C2,C3,C6,S4'.split(','), 
                    #unfinished - special groups
                    'Ih' :  ''.split(','),
                    'T'  :  ''.split(','),
                    'Th' :  ''.split(','),
                    'Td' :  ''.split(','),
                    'O'  :  ''.split(','),
                    'Oh' :  ''.split(','),
                    'I ' :  ''.split(','),
                    'Civ':  ''.split(','),
                    'Div':  ''.split(',')}
                    
point_group_dict = {'C1' :  'ident,ident'.split(','),
                    'Cs' :  'sig_h,ident'.split(','),
                    'Ci' :  'i,ident'.split(','),
                    'C2' :  'C2,ident'.split(','),
                    'C3' :  'C3,ident,C3^2'.split(','),
                    'C4' :  'C4,ident,C2,C4^3'.split(','),
                    'C5' :  'C5,ident,C5^2,C5^3,C5^4'.split(','),
                    'C6' :  'C6,ident,C3,C2,C3^2,C6^5'.split(','),
                    'C7' :  'C7,ident,C7^2,C7^3,C7^4,C7^5,C7^6'.split(','),
                    'C8' :  'C8,ident,C4,C8^3,C2,C8^5,C4^3,C8^7'.split(','),
                    'D2' :  'C2,C2*,C2**'.split(','),
                    'D3' :  '2C3,3C2*'.split(','),
                    'D4' :  '2C4,2C2*,C2,2C2**'.split(','),
                    'D5' :  '2C5,5C2*,2C5^2'.split(','),
                    'D6' :  '2C6,3C2*,2C3,C2,3C2*,3C2**'.split(','),
                    'D7' :  '2C7,2C7^2,2C7^3,7C2*'.split(','),
                    'D8' :  '2C8,4C2*,2C4,2C8^3,C2,4C2**'.split(','),
                    'C2v':  'C2,sig_v(xz),sig_v(yz)'.split(','),
                    'C3v':  '2C3,3sig_v'.split(','),
                    'C4v':  '2C4,2sig_v,C2,2sig_d'.split(','),
                    'C5v':  '2C5,5sig_v,2C5^2'.split(','),
                    'C6v':  '2C6,3sig_v,2C3,C2,3sig_d'.split(','),
                    'C7v':  '2C7,7sig_v,2C7^2,2C7^3'.split(','),
                    'C8v':  '2C8,4sig_v,2C4,2C8^3,C2,4sig_d'.split(','),
                    'C2h':  'C2,i,sig_h'.split(','),
                    'C3h':  'C3,ident,C3^2,sig_h,S3,S3^5'.split(','),
                    'C4h':  'C4,ident,C2,C4^3,i,S4^3,sig_h,S4'.split(','),
                    'C5h':  'C5,ident,C5^2,C5^3,C5^4,sig_h,S5,S5^7,S5^3,S5^9'.split(','),
                    'C6h':  'C6,ident,C3,C2,C3^2,C6^5,i,S3^5,S6^5,sig_h,S6,S3'.split(','),
                    'D2h':  'C2,C2*,C2**,i,sig_h,sig_v(xz),sig_v(yz)'.split(','),
                    'D3h':  '2C3,3C2*,sig_h,2S3,3sig_v'.split(','),
                    'D4h':  '2C4,2C2*,C2,2C2**,i,2S4,sig_h,2sig_v,2sig_d'.split(','),
                    'D5h':  '2C5,5C2*,2C5^2,sig_h,2S5,2S5^3,5sig_v'.split(','),
                    'D6h':  '2C6,3C2*,2C3,C2,3C2**,i,2S3,2S6,sig_h,3sig_d,3sig_v'.split(','),
                    'D7h':  '2C7,7C2*,2C7^2,2C7^3,sig_h,2S7,2S7^3,2S7^5,7sig_v'.split(','),
                    'D8h':  '2C8,4C2*,2C4,2C8^3,C2,4C2**,i,2S8^3,2S8,2S4,sig_h,4sig_v,4sig_d'.split(','),
                    'D2d':  '2S4,C2*,C2**,C2,2sig_d'.split(','),   
                    'D3d':  '2C3,3C2*,i,2S6,3sig_d'.split(','),
                    'D4d':  '2S8,4C2*,2C4,2S8^3,C2,4sig_d'.split(','),
                    'D5d':  'C5,5C2*,C5^2,i,S10,S10^3,S10^7,S10^9,5sig_d'.split(','),
                    'D6d':  '2S12,6C2*,2C6,2S4,2C3,2S12^5,C2,6sig_d'.split(','),
                    'D7d':  '2C7,7C2*,2C7^2,2C7^3,i,2S14^5,2S14^3,2S14,7sig_d'.split(','),
                    'D8d':  '2S16,8C2*,2C8,2S16^3,2C4,2S16^5,2C8^3,2S16^7,C2,8sig_d'.split(','),
                    'S2' :  'i,ident'.split(','),
                    'S4' :  'S4,ident,C2,S4^3'.split(','),
                    'S6' :  'C3,ident,C3^2,i,S6^5,S6'.split(','),
                    'S8' :  'S8,ident,C4,S8^3,C2,S8^5,C4^3,S8^7'.split(','),
                    'S10':  'C5,ident,C5^2,C5^3,C5^4,i,S10^7,S10^9,S10,S10^3'.split(','), 
                    'S12':  'S12,ident,C6,S4,C3,S12^5,C2,S12^7,C3^2,S4^3,C6^5,S12^11'.split(','),
                    'Td' :  'ident,8C31,8C32,8C33,8C34,8C35,8C36,8C37,8C38,3C21,3C22,3C23,6S41,6S42,6S43,6S44,6S45,6S46,6sigd1,6sigd2,6sigd3,6sigd4,6sigd5,6sigd6'.split(','),
                    'Oh' :  'ident,8C31,8C32,8C33,8C34,8C35,8C36,8C37,8C38,6C21,6C22,6C23,6C24,6C25,6C26,6C41,6C42,6C43,6C44,6C45,6C46,C2,2C2*1,2C2*2,i,6S41,6S42,6S43,6S44,6S45,6S46,8S61,8S62,8S63,8S64,8S65,8S66,8S67,8S68,sig_h,2sig_v1,2sig_v2,6sigd1,6sigd2,6sigd3,6sigd4,6sigd5,6sigd6'.split(','),
                  }
           #

point_group_lens = {'C1' : 1, 'Cs' : 2, 'Ci' : 2,  'C2' : 2, 'C3' : 3, 'C4' : 4, 'C5' : 5, 'C6' : 6, 'C7' : 7, 'C8' : 8,  'D2' : 4, 'D3' : 6, 'D4' : 8, 'D5' : 10, 'D6' : 12, 'D7' : 14, 'D8' : 16,  'C2v': 4, 'C3v': 6, 'C4v': 8, 'C5v': 10, 'C6v': 12, 'C7v': 14, 'C8v': 16,  'C2h': 4, 'C3h': 6, 'C4h': 8, 'C5h': 10, 'C6h': 12,  'D2h': 8, 'D3h': 12, 'D4h': 16, 'D5h': 20, 'D6h': 24, 'D7h': 28, 'D8h': 32,  'D2d': 8, 'D3d': 12, 'D4d': 16, 'D5d': 20, 'D6d': 24, 'D7d': 28, 'D8d': 32,  'S2' : 2, 'S4' : 4, 'S6' : 6, 'S8' : 8, 'S10': 10, 'S12': 12, 'T'  :  99, 'Th' :  99, 'Td' :  99, 'O'  :  99, 'Oh' :  99, 'I ' :  99, 'Ih' :  99, 'Civ':  99, 'Div':  99}
                    #not certain this is important for anything

ordered_ops_dict = {'C1':['ident'],'Cs':['ident','sig_h'],'Ci':['ident','i'],'C2':['ident','C2'],
                  'C2v':'ident,C2,sig_v(xz),sig_v(yz)'.split(','),
                  'C3v':'ident,2C3,3sig_v1,3sig_v2,3sig_v3'.split(','),
                  'C4v':'ident,2C4,C2,2sig_v1,2sig_v2,2sig_d1,2sig_d2'.split(','),
                  'C2h':'ident,C2,i,sig_h'.split(','),
                  'D2d':'ident,2S4,C2,2C2*1,2C2*2,2sig_d1,2sig_d2'.split(','),
                  'D2h': 'ident,C2,C2*,C2**,i,sig_h,sig_v(yz),sig_v(xz)'.split(','),
                  'D3d':'ident,2C3,2C2*1,2C2*3,2C2*4,i,S6,2sig_d3,2sig_d4,2sig_d5'.split(','),
                  'D3h':'ident,2C3,2C2*1,2C2*3,2C2*4,sig_h,2S3,3sig_v1,3sig_v2,3sig_v3'.split(','),
                  'D4h':'ident,2C4,C2,2C2*1,2C2*2,2C2**1,2C2**2,i,2S4,sig_h,2sig_v1,2sig_v2,2sig_d1,2sig_d2'.split(','),
                  'D4d':'ident,2S8,2C4,2S8^3,C2,2C2*1,2C2**1,2C2*2,2C2**2,4sig_d1,4sig_d2,4sig_d3,4sig_d4'.split(','),
                  'D5h':'ident,2C5,2C5^2,5C2*1,5C2*2,5C2*3,5C2*4,5C2*5,sig_h,2S5,2S5^3,5sig_v1,5sig_v2,5sig_v3,5sig_v4,5sig_v5'.split(','),
                  'D6h':'ident,2C6,2C3,C2,3C2*1,3C2*2,3C2*3,3C2**1,3C2**2,3C2**3,i,2S3,2S6,sig_h,3sig_d1,3sig_d2,3sig_d3,3sig_v1,3sig_v2,3sig_v3'.split(','),
                  'D7h':'ident,2C7,2C72,2C73,7C2*1,7C2*2,7C2*3,7C2*4,7C2*5,7C2*6,7C2*7,sig_h,2S71,2S73,2S75,7sig_v1,7sig_v2,7sig_v3,7sig_v4,7sig_v5,7sig_v6,7sig_v7'.split(','),
                  'D8h':'ident,2C8,2C8^3,2C4,C2,4C2*1,4C2*2,4C2*3,4C2*4,4C2**1,4C2**2,4C2**3,4C2**4,i,2S8,2S8^3,2S4,sig_h,4sig_d1,4sig_d2,4sig_d3,4sig_d4,4sig_v1,4sig_v2,4sig_v3,4sig_v4'.split(','),
                  'Td' :'ident,8C31,8C32,8C33,8C34,8C35,8C36,8C37,8C38,3C21,3C22,3C23,6S41,6S42,6S43,6S44,6S45,6S46,6sigd1,6sigd2,6sigd3,6sigd4,6sigd5,6sigd6'.split(','),
                  'Oh' :'ident,8C31,8C32,8C33,8C34,8C35,8C36,8C37,8C38,6C21,6C22,6C23,6C24,6C25,6C26,6C41,6C42,6C43,6C44,6C45,6C46,3C21,3C22,3C23,i,6S41,6S42,6S43,6S44,6S45,6S46,8S61,8S62,8S63,8S64,8S65,8S66,8S67,8S68,sig_h,2sig_v1,2sig_v2,6sigd1,6sigd2,6sigd3,6sigd4,6sigd5,6sigd6'.split(','),
                   }

pgt_dict ={'C1':{'A':[1]},
           'Cs':{"A'":[1,1],"A''":[1,0]},
           'Ci':{'Ag':[1,1],'Au':[1,0]},
           'C2':{'A':[1,1],'B':[1,0]},
           'C2v':{'A1' :[1,1,1,1], 
                  'A2' :[1,1,0,0], 
                  'B1' :[1,0,1,0], 
                  'B2' :[1,0,0,1]},
           'C3v':{'A1' :[1,1,1,1,1], 
                  'A2' :[1,1,0,0,0], 
                  'E1' :[1,0,1,0,0], 
                  'E2' :[1,0,0,1,0], 
                  'E3' :[1,0,0,0,1]},

           'C2h':{'Ag' :[1,1,1,1], 
                  'Bg' :[1,0,1,0], 
                  'Au' :[1,1,0,0], 
                  'Bu' :[1,0,0,1]},
           'C4v':{'A1'  :[1,1,1,1,1,1,1], 
                  'A2'  :[1,1,1,0,0,0,0], 
                  'B1'  :[1,0,1,1,1,0,0],  
                  'B2'  :[1,0,1,0,0,1,1], 
                  'Ex'  :[1,0,0,1,0,0,0], 
                  'Ey'  :[1,0,0,0,1,0,0], 
                  'Ex+y':[1,0,0,0,0,1,0], 
                  'Ex-y':[1,0,0,0,0,0,1]},
           'D2h':{'Ag' : [1,1,1,1,1,1,1,1],
                  'B1g': [1,1,0,0,1,1,0,0],
                  'B2g': [1,0,1,0,1,0,1,0],
                  'B3g': [1,0,0,1,1,0,0,1],
                  'Au' : [1,1,1,1,0,0,0,0],
                  'B1u': [1,1,0,0,0,0,1,1],
                  'B2u': [1,0,1,0,0,1,0,1],
                  'B3u': [1,0,0,1,0,1,1,0]},
          'D2d': {'A1'   :[1,1,1,1,1,1,1],
                  'A2'   :[1,1,1,0,0,0,0],
                  'B1'   :[1,0,1,1,1,0,0],
                  'B2'   :[1,0,1,0,0,1,1],
                  'Ex'   :[1,0,0,1,0,0,0],
                  'Ey'   :[1,0,0,0,1,0,0],
                  'Ex+y' :[1,0,0,0,0,1,0],
                  'Ex-y' :[1,0,0,0,0,0,1]},
          'D3h': {"A'1"  :[1,1,1,1,1,1,1,1,1,1],
                  "A'2"  :[1,1,0,0,0,1,1,0,0,0],
                  "A''1" :[1,1,1,1,1,0,0,0,0,0],
                  "A''2" :[1,1,0,0,0,0,0,1,1,1],
                  "E'1"  :[1,0,1,0,0,1,0,1,0,0],
                  "E'2"  :[1,0,0,1,0,1,0,0,1,0],
                  "E'3"  :[1,0,0,0,1,1,0,0,0,1],
                  "E''1" :[1,0,1,0,0,0,0,0,0,0],
                  "E''2" :[1,0,0,1,0,0,0,0,0,0],
                  "E''3" :[1,0,0,0,1,0,0,0,0,0]},
                  #       E C3C2C2C2i S6ovovov
           'D3d': {'A1g' :[1,1,1,1,1,1,1,1,1,1],
                   'A2g' :[1,1,0,0,0,1,1,0,0,0],
                   'A1u' :[1,1,1,1,1,0,0,0,0,0],
                   'A2u' :[1,1,0,0,0,0,0,1,1,1],
                   'Eg1' :[1,0,1,0,0,1,0,1,0,0],
                   'Eg2' :[1,0,0,1,0,1,0,0,1,0],
                   'Eg3' :[1,0,0,0,1,1,0,0,0,1],
                   'Eu1' :[1,0,1,0,0,0,0,0,0,0],
                   'Eu2' :[1,0,0,1,0,0,0,0,0,0],
                   'Eu3' :[1,0,0,0,1,0,0,0,0,0],
                   'Eu4' :[1,0,0,0,0,0,0,1,0,0],
                   'Eu5' :[1,0,0,0,0,0,0,0,1,0],
                   'Eu6' :[1,0,0,0,0,0,0,0,0,1]},
          'D4h': {'A1g'   : [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                  'A2g'   : [1,1,1,0,0,0,0,1,1,1,0,0,0,0],
                  'B1g'   : [1,0,1,1,1,0,0,1,0,1,1,1,0,0],
                  'B2g'   : [1,0,1,0,0,1,1,1,0,1,0,0,1,1],
                  'A1u'   : [1,1,1,1,1,1,1,0,0,0,0,0,0,0],
                  'A2u'   : [1,1,1,0,0,0,0,0,0,0,1,1,1,1],
                  'B1u'   : [1,0,1,1,1,0,0,0,1,0,0,0,1,1],
                  'B2u'   : [1,0,1,0,0,1,1,0,1,0,1,1,0,0],
                  'Egx'   : [1,0,0,1,0,0,0,1,0,0,0,1,0,0],
                  'Egy'   : [1,0,0,0,1,0,0,1,0,0,1,0,0,0],
                  'Egx+y' : [1,0,0,0,0,1,0,1,0,0,0,0,1,0],
                  'Egx-y' : [1,0,0,0,0,0,1,1,0,0,0,0,0,1],
                  'Eux'   : [1,0,0,1,0,0,0,0,0,1,1,0,0,0],
                  'Euy'   : [1,0,0,0,1,0,0,0,0,1,0,1,0,0],
                  'Eux+y' : [1,0,0,0,0,1,0,0,0,1,0,0,0,1],
                  'Eux-y' : [1,0,0,0,0,0,1,0,0,1,0,0,1,0]}, 
                   #       E S8C4S8C2C2C2C2C2odododod
          'D4d': {'A1'   : [1,1,1,1,1,1,1,1,1,1,1,1,1],#D2d
                  'A2'   : [1,1,1,1,1,0,0,0,0,0,0,0,0],#S8
                  'B1'   : [1,0,1,0,1,1,1,1,1,0,0,0,0],#D4
                  'B2'   : [1,0,1,0,1,0,0,0,0,1,1,1,1],#C4v
                  'E1a'   :[1,0,0,0,1,1,0,1,0,0,0,0,0],#D2
                  'E1b'   :[1,0,0,0,1,0,1,0,1,0,0,0,0],#D2
                  'E2a'   :[1,0,0,0,1,0,0,0,0,1,0,1,0],#C2v
                  'E2b'   :[1,0,0,0,1,0,0,0,0,0,1,0,1],#C2v
                  'E3a'   :[1,0,0,0,0,1,0,0,0,0,0,0,0],#C2
                  'E3b'   :[1,0,0,0,0,0,1,0,0,0,0,0,0],#C2
                  'E3c'   :[1,0,0,0,0,0,0,1,0,0,0,0,0],#C2
                  'E3d'   :[1,0,0,0,0,0,0,0,1,0,0,0,0],#C2
                  'E3e'   :[1,0,0,0,0,0,0,0,0,1,0,0,0],#Cs
                  'E3f'   :[1,0,0,0,0,0,0,0,0,0,1,0,0],#Cs
                  'E3g'   :[1,0,0,0,0,0,0,0,0,0,0,1,0],#Cs
                  'E3h'   :[1,0,0,0,0,0,0,0,0,0,0,0,1],#Cs
                  },
          'D5h':{   "A'1":  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "A'2":  [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    "A''1": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    "A''2": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    "E'1a": [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],   
                    "E'1b": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],   
                    "E'1c": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],   
                    "E'1d": [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],   
                    "E'1e": [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                    "E'2a": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],   
                    "E'2b": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    "E'2c": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    "E'2d": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    "E'2e": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    "E''1a":[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "E''1b":[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "E''1c":[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "E''1d":[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "E''1e":[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    "E''2a":[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   
                    "E''2b":[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    "E''2c":[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    "E''2d":[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    "E''2e":[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                         },
                         #  E C6 C3 C2 C2'C2'C2'C2"C2"C2"i  S3 S6 oh od1od2od3ov1ov2ov3
         'D6h': { 'A1g'   : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  'A2g'   : [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                  'B1g'   : [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                  'B2g'   : [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                  'A1u'   : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'A2u'   : [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                  'B1u'   : [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
                  'B2u'   : [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
                  'E2g1'  : [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],#d2h
                  'E2g2'  : [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],#d2h
                  'E2g3'  : [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],#d2h
                  'E1g1'  : [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],#c2h
                  'E1g2'  : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],#c2h
                  'E1g3'  : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],#c2h
                  'E1g4'  : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],#c2h
                  'E1g5'  : [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],#c2h
                  'E1g6'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],#c2h
                  'E1u1'  : [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],#c2v
                  'E1u2'  : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],#c2v
                  'E1u3'  : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],#c2v
                  'E1u4'  : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],#c2v
                  'E1u5'  : [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],#c2v
                  'E1u6'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],#c2v
                  'E2u1'  : [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#d2
                  'E2u2'  : [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#d2
                  'E2u3'  : [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#d2
                  'E2u4'  : [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                  'E2u5'  : [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                  'E2u6'  : [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],},   
          'D7h':{"A1'"   :[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  "A1''"  :[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "A2'"   :[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  "A2''"  :[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  "E'1"   :[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  "E'2"   :[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  "E'3"   :[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  "E'4"   :[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  "E'5"   :[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  "E'6"   :[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  "E'7"   :[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  "E''1"  :[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "E''2"  :[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "E''3"  :[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "E''4"  :[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "E''5"  :[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "E''6"  :[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "E''7"  :[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},

#                         'E  2C82C82C4C2 C2'C2'C2'C2'C2"C2"C2"C2" i 2S82S82S4 sh,sd sd sd sd sv sv sv sv
          'D8h': {"A1g"  : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  "A2g"  : [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],#': 'C8h', 
                  "B1g"  : [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],#': 'D4h', 
                  "B2g"  : [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],#': 'D4h', 
                  "E2g1" : [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],#: 'D2h', 
                  "E2g2" : [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],#: 'D2h', 
                  "E2g3" : [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],#: 'D2h', 
                  "E2g4" : [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],#: 'D2h', 
                  "E1g1" : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],#: 'C2h', 
                  "E1g2" : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],#: 'C2h', 
                  "E1g3" : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],#: 'C2h', 
                  "E1g4" : [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],#: 'C2h', 
                  "E3g1" : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],#: 'C2h', 
                  "E3g2" : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],#: 'C2h', 
                  "E3g3" : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],#: 'C2h', 
                  "E3g4" : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],#: 'C2h', 
                  'A1u'  : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#': 'D8',
                  'A2u'  : [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],#': 'C8v', 
                  'B1u'  : [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],#': 'D4d', 
                  'B2u'  : [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],#': 'D4d', 
                  'E2u1' : [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],#: 'D2d', 
                  'E2u2' : [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],#: 'D2d', 
                  'E2u3' : [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],#: 'D2d', 
                  'E2u4' : [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],#: 'D2d', 
                  'E1u1' : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],#: 'C2v', 
                  'E1u2' : [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],#: 'C2v', 
                  'E1u3' : [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],#: 'C2v', 
                  'E1u4' : [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],#: 'C2v', 
                  'E3u1' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],#: 'C2v', 
                  'E3u2' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],#: 'C2v', 
                  'E3u3' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],#: 'C2v', 
                  'E3u4' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],#: 'C2v', 
# from the functions in the point group table - technically the E1 and E3 are identical in terms or residual proper symmetry
# operations, as there's no formal 3-pole (or 6-pole) in the 8-fold group, but these can be separated by an 
# interested party later by manual delineation of the E1g and E3g principal components: Here's the assignment as
# written:  
# A1g  - to D8h
# A2g  - to C8h
# B1g  - to D4h
# B2g  - to D4h
# A1u  - to D8
# A2u  - to C8v
# B1u  - to D4d
# B2u  - to D4d
# E1g  - to C2h (B1 axis)
# E2g  - to D2h
# E3g  - to C2h (B2 axis)
# E1u  - to C2v (B1 axis)
# E2u  - to D2d
# E3u  - to C2v (B2 axis)

          },
           # aligned with S4 axis (i.e. nodes along 111,1-1-1,-11-1 and -1-11)
                 #      Td   E  C31C32C33C34C35C36C37C38C2zC2xC2yS41S42S43S44S45S46sd1sd2sd3sd4sd5sd6
          'Td' : {'A1'    : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                  'A2'    : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  'Ez'    : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0], 
                  'Ex'    : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1], 
                  'Ey'    : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], 
                  'T1(a)' : [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                  'T1(b)' : [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                  'T1(c)' : [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
                  'T1(d)' : [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                  'T2(a)' : [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'T2(b)' : [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'T2(c)' : [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'T2(d)' : [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
                  #           E C31C32C33C34C35C36C37C38C21C22C23C24C25C26C41C42C43C44C45C46C2zC2xC2y i S41S42S43S44S45S46S61S62S63S64S65S66S67S68sh1sh2sh3sd1sd2sd3sd4sd5sd6
                  #             xyz-yz--zx-zxy--y----x-- xz yz -xz-yzxy-xy z -z  x -x  y -y  z  x  y    xz yz -xz-yzxy-xy xyz-yz--zx-zxy--y----x-- z  x  y  xy-xy xz-xz yz-yz
#      'Oh' : {'A1g'    : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], #oh
#              'A2g'    : [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], #Th
#              'Egx'    : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1], #D4h
#              'Egy'    : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0], #D4h
#              'Egz'    : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0], #D4h
#              'T1g(a)' : [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1], #D3d 
#              'T1g(b)' : [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1], #D3d 
#              'T1g(c)' : [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0], #D3d 
#              'T1g(d)' : [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0], #D3d 
#              'T2g(a)' : [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #S6
#              'T2g(b)' : [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #S6
#              'T2g(c)' : [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #S6
#              'T2g(d)' : [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #S6 
#              'A1u'    : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#              'A2u'    : [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], 
#              'Eux'    : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], #D4h
#              'Euy'    : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0], #D4h
#              'Euz'    : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], #D4h
#              'T1u(a)' : [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], #C4h 
#              'T1u(b)' : [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], #C4h
#              'T1u(c)' : [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], #C4h 
#              'T1u(d)' : [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], #C4h 
#              'T2u(a)' : [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1], #S6
#              'T2u(b)' : [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1], #S6
#              'T2u(c)' : [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0], #S6
#              'T2u(d)' : [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0], #S6 {C4h}?}
 #                 E C31C32C33C34C35C36C37C38C21C22C23C24C25C26C41C42C43C44C45C46C2zC2xC2y i S41S42S43S44S45S46S61S62S63S64S65S66S67S68sh1sh2sh3sd1sd2sd3sd4sd5sd6
 #                   xyz-yz--zx-zxy--y----x-- xz yz -xz-yzxy-xy z -z  x -x  y -y  z  x  y    xz yz -xz-yzxy-xy xyz-yz--zx-zxy--y----x-- z  x  y  xy-xy xz-xz yz-yz
'Oh' : {'A1g'   : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],# 'Oh',
        'A2g'   : [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],# 'Th',
        'A1u'   : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],# 'O',
        'A2u'   : [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],# 'Td',
        'T1gx'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],# 'D4h',
        'T1gy'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0],# 'D4h',
        'T1gz'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],# 'D4h',
        'T2gx'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],# 'D2h',
        'T2gy'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],# 'D2h',
        'T2gz'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],# 'D2h',
        'T2ga'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],# 'D2h',
        'Eux'   : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],# 'C4v',
        'Euy'   : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0],# 'C4v',
        'Euz'   : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],# 'C4v',
        'T1ux'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],# 'C4h',
        'T1uy'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],# 'C4h',
        'T1uz'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],# 'C4h',
        'T2ux'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],# 'D2d',
        'T2uy'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],# 'D2d',
        'T2uz'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]}# 'D2d'
}

#        'T1ux'   : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],# 'C2v',
#        'T1ux'   : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],# 'C2v',
#        'T1ux'   : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],# 'C2v',

#Oh equivalent, I think. There's still some uncertainty that I've got the right labels here, and they're still backwards in the actual
#runtime. This is getting far too complicated.
#A1g Oh (sphere)
#A2g Th (volleyball)
#Eg   D3d (D4h / D2h)
#T1g  D4h (C4h)
#T2g  C4h (D2h)
#A1u O 
#A2u Td (tetrahedron)
#Eu  C2v (C4v)
#T1u C4v (C2v)
#T2u D2d (D2d)

symm_multiplicity_tables = {'D2d': [0,1,1,1,1/2,1/2,1/2,1/2,0],
                            'C3v': [0,1,2/3,2/3,2/3,0],
                            'C4v': [0,1,1,1,1/2,1/2,1/2,1/2,0],
                            'D4d': [0,1,1,1,1/2,1/2,1/2,1/2,1/4,1/4,1/4,1/4,1/4,1/4,1/4,1/4,0],
                            'D3h': [0,1,1,1,2/3,2/3,2/3,2/3,2/3,2/3,0],
                            'D3d': [0,1,1,1,2/3,2/3,2/3,2/3,2/3,2/3,0],
                            'D5h': [0,1,1,1,1/5,1/5,1/5,1/5,1/5,1/5,1/5,1/5,1/5,1/5,1/5,1/5,1/5,1/5,1/5,1/5,1/5,1/5,1/5,1/5,0],
                            'D6h': [0,1,1,1,1,1,1,1,2/3,2/3,2/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3,0],
                            'D7h': [0,1,1,1,2/7,2/7,2/7,2/7,2/7,2/7,2/7,2/7,2/7,2/7,2/7,2/7,2/7,2/7,0],
                            'D8h': [0,1,1,1,1/2,1/2,1/2,1/2,1/4,1/4,1/4,1/4,1/4,1/4,1/4,1/4,1,1,1,1,1/2,1/2,1/2,1/2,1/4,1/4,1/4,1/4,1/4,1/4,1/4,1/4,0],
                            'Td': [0,1,2/3,2/3,2/3,3/4,3/4,3/4,3/4,3/4,3/4,3/4,3/4,0],
                            'Oh': [0,1,1/2,1/2,1/2,1/2,1,1,1,1/2,1/2,1/2,1/2,1/2,1/2,1,1,1/3,1/3,1/3,1/3,1/3,1/3,1,1,1,1,1,1,0],
                                }

#e_group_subdict = {'D2d':{'Ex':'ident,C2*,sig_v(xz)'.split(','),
#                          'Ey':'ident,C2**,sig_v(yz)'.split(',')}
#                  }
e_group_parent  = {'C4v':{'Ex':'B1','Ey':'B1','Ex+y':'B2','Ex-y':'B2'},
                   'D4h':{'Egx':'B1g','Egy':'B1g','Egx+y':'B2g','Egx-y':'B2g',
                          'Eux':'B1g','Euy':'B1g','Eux+y':'B2g','Eux-y':'B2g'},
                   'D2d':{'Ex':'B1', 'Ey':'B1','Ex+y':'B2', 'Ex-y':'B2'},
                  }
e_group_parent_multi  = {
                  'C3v':{},
                  'D3h': {"E''1":["E'1","A''1"], "E''2":["E'2","A''1"], "E''3":["E'3","A''1"]},
                  'D3d': {"Eu1":["Eg1","A1u"],"Eu2":["Eg2","A1u"],"Eu3":["Eg3","A1u"], "Eu4":["Eg1","A2u"], "Eu5":["Eg2","A2u"],"Eu6":["Eg3","A2u"]},
                  'D4d': {'E1a':['B1'],'E1b':['B1'],'E2a':['B2'],'E2b':['B2'],'E3a':['E1a','B1'],'E3b':['E1b','B1'],'E3c':['E1a','B1'],'E3d':['E1b','B1'],'E3e':['E2a','B2'],'E3f':['E2b','B2'],'E3g':['E2a','B2'],'E3h':['E2b','B2']},
                  'D5h':{"E'2a" : ["E'1a", "A'2"], "E'2b" : ["E'1b", "A'2"], "E'2c" : ["E'1c", "A'2"], "E'2d" : ["E'1d", "A'2"],  "E'2e" : ["E'1e", "A'2"],  "E''1a": ["E'1a", "A''1"], "E''1b": ["E'1b", "A''1"],  "E''1c": ["E'1c", "A''1"],  "E''1d": ["E'1d", "A''1"],  "E''1e": ["E'1e", "A''1"],  "E''2a": ["E'1a", "A''2"],  "E''2b": ["E'1b", "A''2"],  "E''2c": ["E'1c", "A''2"],  "E''2d": ["E'1d", "A''2"],  "E''2e": ["E'1e", "A''2"]},
                  'D6h': {"E1g1":['E2g1', 'B1g'],"E1g2":['E2g2', 'B1g'],"E1g3":['E2g3', 'B1g'],"E1g4":['E2g1', 'B2g'],"E1g5":['E2g2', 'B2g'],"E1g6":['E2g3', 'B2g'],"E1u1":['E2g1', 'B1u'],"E1u2":['E2g2', 'B1u'],"E1u3":['E2g3', 'B1u'],"E1u4":['E2g1', 'B2u'],"E1u5":['E2g2', 'B2u'],"E1u6":['E2g3', 'B2u'],"E2u1":['E2g1', 'A1u'],"E2u2":['E2g2', 'A1u'],"E2u3":['E2g3', 'A1u'],"E2u4":['E2g1', 'A2u'],"E2u5":['E2g2', 'A2u'],"E2u6":['E2g3', 'A2u']},
                  'Td': {"T2(a)":['T1(a)', 'A2'], "T2(b)":['T1(b)', 'A2'], "T2(c)":['T1(c)', 'A2'], "T2(d)":['T1(d)', 'A2']}, 
                  'D7h':{"E''1": ["E'1", "A1''"],"E''2": ["E'2", "A1''"],"E''3": ["E'3", "A1''"],"E''4": ["E'4", "A1''"],"E''5": ["E'5", "A1''"],"E''6": ["E'6", "A1''"],"E''7": ["E'7", "A1''"]},
                  'D8h':{'E2g1': ['B1g'], 'E2g2': ['B1g'], 'E2g3': ['B2g'], 'E2g4': ['B2g'], 'E1g1': ['B1g', 'E2g1'], 'E1g2': ['B1g', 'E2g2'], 'E1g3': ['B1g', 'E2g1'], 'E1g4': ['B1g', 'E2g2'], 'E3g1': ['B2g', 'E2g3'], 'E3g2': ['B2g', 'E2g4'], 'E3g3': ['B2g', 'E2g3'], 'E3g4': ['B2g', 'E2g4'], 'E2u1': ['B1g'], 'E2u2': ['B1g'], 'E2u3': ['B2g'], 'E2u4': ['B2g'], 'E1u1': ['B1g', 'E2g1'], 'E1u2': ['B1g', 'E2g2'], 'E1u3': ['B1g', 'E2g1'], 'E1u4': ['B1g', 'E2g2'], 'E3u1': ['B2g', 'E2g3'], 'E3u2': ['B2g', 'E2g4'], 'E3u3': ['B2g', 'E2g3'], 'E3u4': ['B2g', 'E2g4']},
                 'Oh': {'T2gx1': ['T1gx'], 'T2gy1': ['T1gy'], 'T2gz1': ['T1gz'], 'T2gx2': ['T1gx'], 'T2gy2': ['T1gy'], 'T2gz2': ['T1gz'], 'Eu1': ['T1gy', 'T2gy2'], 'Eu2': ['T1gx', 'T2gx2'], 'Eu3': ['T1gy', 'T2gy2'], 'Eu4': ['T1gx', 'T2gx2'], 'Eu5': ['T1gz', 'T2gz2'], 'Eu6': ['T1gz', 'T2gz2'], 'T1ux': ['T1gx'], 'T1uy': ['T1gy'], 'T1uz': ['T1gz'], 'T2ux': ['T1gx'], 'T2uy': ['T1gy'], 'T2uz': ['T1gz']}
                 }
        
e_group_unique_sets = {'D2d': [['Ex','Ey','Ex+y','Ex-y']],
'C3v': [['E1'],['E2'],['E3']],
'C4v': [['Ex','Ey'],['Ex+y','Ex-y']],
'D3d': [['Eg1','Eu1','Eu4'],['Eg2','Eu2','Eu5'],['Eg3','Eu3','Eu6']],
'D4d': [['E1a','E2a'],['E3a','E3c'],['E3e','E3g'],['E1b','E2b'],['E3b','E3d'],['E3f','E3h']],
'D3h': [["E'1","E''1"],["E'2","E''2"],["E'3","E''3"]],
'D5h': [["E'1"+x,"E'2"+x,"E''1"+x,"E''2"+x] for x in ['a','b','c','d','e']],
'D6h': ['E2g1,E1g1,E1g4,E1u1,E1u4,E2u1,E2u4'.split(','), 'E2g2,E1g2,E1g5,E1u2,E1u5,E2u2,E2u5'.split(','), 'E2g3,E1g3,E1g6,E1u3,E1u6,E2u3,E2u6'.split(',')],
'D8h': ['E2g1,E1g1,E1g3'.split(','),'E2g2,E1g2,E1g4'.split(','),'E2g3,E3g1,E3g3'.split(','),'E2g4,E3g2,E3g4'.split(','),'E2u1,E1u1,E1u3'.split(','),'E2u2,E1u2,E1u4'.split(','),'E2u3,E3u1,E3u3'.split(','),'E2u4,E3u2,E3u4'.split(',')],
'Oh': [['T1gx','T2gx1','T2gx2', 'T1ux','T2ux'],['T1gy', 'T2uy', 'T1uy', 'T2gy1','T2gy2'],['T1gz', 'T2uz','T1uz','T2gz1','T2gz2']],
}

mondrian_lookup_dict =  {'C1':{'[1]':'C1'},
                        'Cs':{'[1, 1]':'Cs','[1, 0]':'C1'},
                        'Ci':{'[1, 1]':'Ci','[1, 0]':'C1'},
                        'C2':{'[1, 1]':'C2','[1, 0]':'C1'},
                        'C2v':{'[1, 0, 0, 0]':'C1', '[1, 1, 0, 0]':'C2', '[1, 0, 1, 0]':'Cs', '[1, 0, 0, 1]':'Cs', '[1, 1, 1, 1]':'C2v', '[1, 1, 1, 0]':'C2v', '[1, 0, 1, 1]':'C2v', '[1, 1, 0, 1]':'C2v'},
                        'C2h':{'[1, 0, 0, 0]':'C1', '[1, 1, 0, 0]':'C2', '[1, 0, 1, 0]':'Ci', '[1, 0, 0, 1]':'Cs', '[1, 1, 1, 1]':'C2h'},
                        'C3v':{'[1, 1, 1, 1, 1]':'C3v', '[1, 1, 0, 0, 0]':'C3', '[1, 0, 1, 0, 0]':'Cs', '[1, 0, 0, 1, 0]':'Cs', '[1, 0, 0, 0, 1]':'Cs', '[1, 0, 0, 0, 0]':'C1'},
                        'C4v':{'[1, 1, 1, 1, 1, 1, 1]':'C4v', '[1, 1, 1, 0, 0, 0, 0]':'C4', '[1, 0, 1, 1, 1, 0, 0]':'C2v', '[1, 0, 1, 0, 0, 1, 1]':'C2v', '[1, 0, 1, 0, 0, 0, 0]':'C2', '[1, 0, 0, 1, 0, 0, 0]':'Cs', '[1, 0, 0, 0, 1, 0, 0]':'Cs', '[1, 0, 0, 0, 0, 1, 0]':'Cs', '[1, 0, 0, 0, 0, 0, 1]':'Cs', '[1, 0, 0, 0, 0, 0, 0]':'C1'},
                        'D2d':{'[1, 1, 1, 1, 1, 1, 1]':'D2d','[1, 1, 1, 0, 0, 0, 0]':'S4', '[1, 0, 1, 1, 1, 0, 0]':'D2', '[1, 0, 1, 0, 0, 1, 1]':'C2v', '[1, 0, 1, 0, 0, 0, 0]':'C2', '[1, 0, 0, 1, 0, 0, 0]':'C2', '[1, 0, 0, 0, 1, 0, 0]':'C2', '[1, 0, 0, 0, 0, 1, 0]':'Cs', '[1, 0, 0, 0, 0, 0, 1]':'Cs', '[1, 0, 0, 0, 0, 0, 0]':'C1'},
                        'D2h':{'[1, 1, 1, 1, 1, 1, 1, 1]':'D2h','[1, 1, 0, 0, 1, 1, 0, 0]':'C2h', '[1, 0, 1, 0, 1, 0, 1, 0]':'C2h','[1, 0, 0, 1, 1, 0, 0, 1]':'C2h', '[1, 1, 1, 1, 0, 0, 0, 0]':'D2', '[1, 1, 0, 0, 0, 0, 1, 1]':'C2v', '[1, 0, 1, 0, 0, 1, 0, 1]':'C2v', '[1, 0, 0, 1, 0, 1, 1, 0]':'C2v','[1, 1, 0, 0, 0, 0, 0, 0]':'C2', '[1, 0, 1, 0, 0, 0, 0, 0]':'C2', '[1, 0, 0, 1, 0, 0, 0, 0]':'C2', '[1, 0, 0, 0, 1, 0, 0, 0]':'Ci', '[1, 0, 0, 0, 0, 1, 0, 0]':'Cs', '[1, 0, 0, 0, 0, 0, 1, 0]':'Cs', '[1, 0, 0, 0, 0, 0, 0, 1]':'Cs', '[1, 0, 0, 0, 0, 0, 0, 0]':'C1'},
                        'D3d':{'[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]':'D3d', '[1, 1, 0, 0, 0, 1, 1, 0, 0, 0]':'S6', '[1, 0, 1, 0, 0, 1, 0, 1, 0, 0]':'C2v', '[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]':'C2v', '[1, 0, 0, 0, 1, 1, 0, 0, 0, 1]':'C2v', '[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]':'D3', '[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]':'C3v', '[1, 0, 1, 0, 0, 0, 1, 1, 0, 0]':'C2h', '[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]':'C2h', '[1, 0, 0, 0, 1, 0, 1, 0, 0, 1]':'C2h','[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'C3','[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'Ci','[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'S6?','[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C1',},
                        'D3h':{'[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]':'D3h',  '[1, 1, 0, 0, 0, 1, 1, 0, 0, 0]':'C3h',  '[1, 0, 1, 0, 0, 1, 0, 1, 0, 0]':'C2v',  '[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]':'C2v',  '[1, 0, 0, 0, 1, 1, 0, 0, 0, 1]':'C2v',  '[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]':'D3',  '[1, 1, 0, 0, 0, 0, 0, 1, 1, 1]':'C3v',  '[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'C3', '[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]':'C2', '[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'C2', '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'C2', '[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'Cs', '[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'???', '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'Cs', '[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'Cs', '[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'Cs', '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C1',} ,
                        'D4d':{'[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]':'D4d','[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'S8','[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]':'D4','[1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1]':'C4v','[1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]':'D2','[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]':'D2','[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]':'C2v','[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]':'C2v','[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'C4','[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C1',},
                        'D4h':{'[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]':'D4h','[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]':'C4h','[1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0]':'D2h','[1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1]':'D2h','[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]':'D4 ','[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]':'C4v','[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0]':'D2d','[1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1]':'D2d','[1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]':'C2h','[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]':'C2h','[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]':'C2h','[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]':'C2h','[1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]':'C2h',         '[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]':'C2v','[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]':'C2v','[1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]':'C2v','[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]':'C2v','[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]':'C2v','[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]':'C2v','[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]':'D2 ','[1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2 ','[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'S4 ','[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C4 ','[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2 ','[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2 ','[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2 ','[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'C2 ','[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]':'C2 ','[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'Ci ','[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'Cs ','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'Cs ','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'Cs ','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'Cs ','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'Cs ','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C1 ',},
                        'D5h':{'[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]':'D5h','[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]':'C5h','[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'D5','[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]':'C5v','[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]':'C2v','[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]':'C2v','[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]':'C2v','[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0]':'C2v','[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]':'C2v','[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]':'Cs','[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'Cs','[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C5','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C1',},
                        'D6h':{'[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]':'D6h','[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]':'C6h','[1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0]':'D3d','[1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1]':'D3d','[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D6', '[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]':'C6v','[1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1]':'D3h','[1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0]':'D3h','[1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]':'D2h','[1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]':'D2h','[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]':'D2h','[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'C2h','[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'C2h','[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'C2h','[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'C2h','[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'C2h','[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'C2h','[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]':'C2v','[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]':'C2v','[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]':'C2v','[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]':'C2v','[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]':'C2v','[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]':'C2v','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]':'C2v','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0]':'C2v','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]':'C2v','[1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2','[1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2','[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2','[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]':'C3h','[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'C2h','[1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D3','[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]':'C3v','[1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D3','[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]':'C3v','[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]':'C2v','[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]':'C2v','[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]':'C2v','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C1','[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C6','[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C3','[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'Ci','[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]':'S6','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'Cs', },   
                        'D7h':{'[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]':'D7h','[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D7','[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]':'C7h','[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]':'C7v','[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'C2v','[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'C2v','[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'C2v','[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'C2v','[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'C2v','[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'C2v','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'C2v','[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'Cs','[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'Cs','[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C7',},
       #      Td   E  C31C32C33C34C35C36C37C38C2zC2xC2yS41S42S43S44S45S46sd1sd2sd3sd4sd5sd6
          'Td' : {'[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]':'Td', 
                  '[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'T', 
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]':'D2d', 
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]':'D2d', 
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]':'D2d', 
                  '[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]':'C3v',
                  '[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]':'C3v',
                  '[1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]':'C3v',
                  '[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]':'C3v',
                  '[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C3',
                  '[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C3',
                  '[1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C3',
                  '[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C3',                  
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'S4',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]':'S4',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'S4',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C1'},
                  #Unfinished! Don't use!
          'Oh' : {
# E C31C32C33C34C35C36C37C38C21C22C23C24C25C26C41C42C43C44C45C46C2zC2xC2y i S41S42S43S44S45S46S61S62S63S64S65S66S67S68sh1sh2sh3sd1sd2sd3sd4sd5sd6
#   xyz-yz--zx-zxy--y----x-- xz yz -xz-yzxy-xy z -z  x -x  y -y  z  x  y    xz yz -xz-yzxy-xy xyz-yz--zx-zxy--y----x-- z  x  y  xy-xy xz-xz yz-yz
'[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]':'Oh', #A1g
'[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]':'Th', #A2g
'[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'O',  #A1u
'[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]':'Td', #A2u
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]':'D4h', #Egx
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]':'D4h', #Egy
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]':'D4h', #Egz
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D4', #x
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D4', #y
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D4', #z
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C4', #x
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C4', #y
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C4', #z
'[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]':'D3d', #a
'[1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]':'D3d', #b
'[1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]':'D3d', #c
'[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]':'D3d', #d
'[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D3', #a
'[1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D3', #b
'[1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D3', #c
'[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D3', #d
'[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C3', #a
'[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C3', #b
'[1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C3', #c
'[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C3', #d
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2', 
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2', 
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2', 
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2', 
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2', 
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2', 
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2', 
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2', 
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2', 
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'Ci', 
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'Cs',
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]':'Cs',
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'Cs',
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'Cs',
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'Cs', 
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'Cs',
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'Cs',
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'Cs',
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'Cs',
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C1',
'[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'T', 
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]':'C4v', #x
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0]':'C4v', #y
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]':'C4v', #z
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]':'C2v', #x
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]':'C2v', #x
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]':'C2v', #y
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]':'C2v', #y
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]':'C2v', #z
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]':'C2v', #z
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]':'C2v', #xz
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]':'C2v', #yz
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]':'C2v', #-xz
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]':'C2v', #-yz
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]':'C2v', #xy
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]':'C2v', #-xy
'[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]':'C3v', #a
'[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]':'C3v', #b
'[1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]':'C3v', #c
'[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]':'C3v', #d
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'C4h', #x
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]':'C4h', #y
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'C4h', #z
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'C2h', #x
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]':'C2h', #y
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'C2h', #z
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'C2h', #xz
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'C2h', #yz
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'C2h', #-xz
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'C2h', #-yz
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'C2h', #xy
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'C2h', #-xy
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]':'D2h', # type 1
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]':'D2h', # type 2x
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]':'D2h', # type 2y
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]':'D2h', # type 2z
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]':'D2d', # type 1x
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]':'D2d', # type 2x
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]':'D2d', # type 1y
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]':'D2d', # type 2y
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]':'D2d', # type 1z
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]':'D2d', # type 2z
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2', # type 1
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2', # type 2x
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2', # type 2y
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2', # type 2z
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'S4', # x
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'S4', # y
'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'S4', # z
'[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'S6',
'[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'S6',
'[1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'S6',
'[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'S6', 
              
                   } ,
#                   E 2C82C82C4C2 C2'C2'C2'C2'C2"C2"C2"C2" i 2S82S82S4 sh,sd sd sd sd sv sv sv sv

          'D8h': {'[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]':'D8h',
                  '[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'S8',
                  '[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'C4h',
                  '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'S4',
                  '[1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1]':'D2d',
                  '[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0]':'D2d',
                  '[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0]':'D2d',
                  '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]':'D2d',
                  '[1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]':'D4d',
                  '[1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]':'D4d',
                  '[1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]':'D4h',
                  '[1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]':'D4h',
                  '[1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]':'D2h',
                  '[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]':'D2h',
                  '[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]':'D2h',
                  '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]':'D2h',
                  '[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'C8h',
                  '[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'C4h!',
                  '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'C2h',
                  '[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'C2h',
                  '[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'C2h',
                  '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'C2h',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'C2h',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]':'C2h',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'C2h',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'C2h',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'C2h',
                  '[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]':'C8v',
                  '[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]':'C4v',
                  '[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]':'C4v',
                  '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]':'C2v',
                  '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]':'C2v',
                  '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]':'C2v',
                  '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]':'C2v',
                  '[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]':'C2v',
                  '[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]':'C2v',
                  '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]':'C2v',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]':'C2v',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]':'C2v',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]':'C2v',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]':'C2v',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]':'C2v',
                  '[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D8',
                  '[1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D4',
                  '[1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D4',
                  '[1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2',
                  '[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2',
                  '[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2',
                  '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2',
                  '[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C8',
                  '[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C4',
                  '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2',
                  '[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2',
                  '[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2',
                  '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'Ci',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'Cs',
                  '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C1',

          },
                    }     

labels_dict = {'C2v':'A1,A2,B1,B2'.split(','),
'C2h':'Ag,Au,Bg,Bu'.split(','),
'C3v':'A1,A2,E'.split(','),
'C4v':'A1,A2,B1,B2,E'.split(','),
'D2h':'Ag,B1g,B2g,B3g,Au,B1u,B2u,B3u'.split(','),
'D3h':"A'1,A'2,E',A''1,A''2,E''".split(','),
'D4h':'A1g,A2g,B1g,B2g,Eg,A1u,A2u,B1u,B2u,Eu'.split(','),
'D6h':'A1g,A2g,B1g,B2g,E1g,E2g,A1u,A2u,B1u,B2u,E1u,E2u'.split(','),
'D8h':'A1g,A2g,B1g,B2g,E1g,E2g,E3g,A1u,A2u,B1u,B2u,E1u,E2u,E3g'.split(','),
'Td' :'A1,A2,E,T1,T2'.split(','),
'Oh' :'A1g,A2g,Eg,T1g,T2g,A1u,A2u,Eu,T1u,T2u'.split(','),
}

labels_dict_ext = {'D4h':{'A1g':'A1g','A2g':'A2g','B1g':'B1g','B2g':'B2g','A1u':'A1u','A2u':'A2u','B1u':'B1u','B2u':'B2u','Egx' :'Eg' ,'Egy' :'Eg' ,'Egx+y' :'Eg' ,'Egx-y' :'Eg' ,'Eux' :'Eu' ,'Euy' :'Eu' ,'Eux+y' :'Eu' ,'Eux-y' :'Eu' },
'C3v':{'A1':'A1','A2':'A2','E1' :'E' ,'E2' :'E' ,'E3' :'E'},
'C4v':{'A1':'A1','A2':'A2','B1':'B1','B2':'B2','Ex' :'E' ,'Ey' :'E' ,'Ex+y' :'E' ,'Ex-y' :'E' },
'D3h': {"A'1":"A'1","A'2":"A'2","A''1":"A''1","A''2":"A''2","E'1":"E'","E'2":"E'","E'3":"E'","E''1":"E''","E''2":"E''","E''3":"E''"},
'D6h': {'A1g':'A1g','A2g':'A2g','B1g':'B1g','B2g':'B2g','A1u':'A1u','A2u':'A2u','B1u':'B1u','B2u':'B2u',"E2g1":"E2g","E2g2":"E2g","E2g3":"E2g","E1u1":"E1u","E1u2":"E1u","E1u3":"E1u","E1u4":"E1u","E1u5":"E1u","E1u6":"E1u","E1g1":"E1g","E1g2":"E1g","E1g3":"E1g","E1g4":"E1g","E1g5":"E1g","E1g6":"E1g","E2u1":"E2u","E2u2":"E2u","E2u3":"E2u","E2u4":"E2u","E2u5":"E2u","E2u6":"E2u"},
'D8h': {'A1g':'A1g','A2g':'A2g','B1g':'B1g','B2g':'B2g','A1u':'A1u','A2u':'A2u','B1u':'B1u','B2u':'B2u',"E1g1":"E1g","E1g2":"E1g","E1g3":"E1g","E1g4":"E1g","E1u1":"E1u","E1u2":"E1u","E1u3":"E1u","E1u4":"E1u","E2g1":"E2g","E2g2":"E2g","E2g3":"E2g","E2g4":"E2g","E2u1":"E2u","E2u2":"E2u","E2u3":"E2u","E2u4":"E2u","E3g1":"E3g","E3g2":"E3g","E3g3":"E3g","E3g4":"E3g","E3u1":"E3u","E3u2":"E3u","E3u3":"E3u","E3u4":"E3u",},
'Oh' : {'A1g'   :'A1g'   , 'A2g'   :'A2g'   , 'Ega'   :'Eg'   , 'Egb'   :'Eg'   , 'Egc'   :'Eg'   , 'Egd'   :'Eg'   , 'T1gx'  :'T1g'  , 'T1gy'  :'T1g'  , 'T1gz'  :'T1g'  , 'T2gx1' :'T2g' , 'T2gy1' :'T2g' , 'T2gz1' :'T2g' , 'T2gx2' :'T2g' , 'T2gy2' :'T2g' , 'T2gz2' :'T2g' , 'A1u'   :'A1u'   , 'A2u'   :'A2u'   , 'Eu1'   :'Eu'   , 'Eu2'   :'Eu'   , 'Eu3'   :'Eu'   , 'Eu4'   :'Eu'   , 'Eu5'   :'Eu'   , 'Eu6'   :'Eu'   , 'T1ux'  :'T1u'  , 'T1uy'  :'T1u'  , 'T1uz'  :'T1u'  , 'T2ux'  :'T2u'  , 'T2uy'  :'T2u'  , 'T2uz'  :'T2u'  }
      }  

mondrian_orientation_dict = {'C1':{'A':'v'},
                            'Cs':{"A'":'v',"A''":'h'},
                            'Ci':{'Ag':'v','Au':'h'},
                            'C2':{'A':'v','B':'h'},
                            'C2v':{'A1':'v', 'A2':'h','B1':'v','B2':'h'},
                            'C3v':{'A1':'v', 'A2':'v','E1':'h','E2':'h','E3':'h'},
                            'C4v':{'A1':'v', 'A2':'h','B1':'v','B2':'h','Ex':'v','Ey':'v','Ex+y':'h','Ex-y':'h',},
                            'C2h':{'Ag':'v', 'Au':'h','Bg':'h','Bu':'v'},
                            'D2d':{'A1':'h', 'A2':'h','B1':'h','B2':'h','Ex':'v','Ey':'v','Ex+y':'v','Ex-y':'v'},
                            'D2h':{'Ag':'v', 'B1g':'v','B2g':'h','B3g':'h','Au':'h', 'B1u':'h','B2u':'v','B3u':'v'},
                            'D3d':{'A1g':'v','A2g':'h','A1u':'v','A2u':'h','Eg1':'v','Eg2':'v','Eg3':'v','Eu1':'h','Eu2':'h','Eu3':'h'},
                            'D3h':{"A'1" :'v', "A'2" :'v',  "A''1" :'h', "A''2" :'h',"E'1" :'v', "E'2" :'v', "E'3" :'v',"E''1" : 'h',"E''2" : 'h',"E''3" : 'h'},
                            'D4d':{'A1':'v', 'A2':'h', 'B1':'v', 'B2':'h', 'E1a':'h', 'E1b':'h', 'E2a':'h', 'E2b':'h', 'E3a':'v','E3b':'v','E3c':'v','E3d':'v','E3e':'v','E3f':'v','E3g':'v','E3h':'v'},
                            'D4h':{'A1g' :'v','A2g' :'v','B1g' :'v','B2g' :'v','A1u' :'h','A2u' :'h','B1u' :'h','B2u' :'h','Egx' :'h','Egy' :'h','Egx+y':'h','Egx-y':'h','Eux' :'v','Euy' :'v','Eux+y':'v','Eux-y':'v'},
                            'D5h':{"A'1":  'v',"A'2":  'v',"E'1a": 'v',"E'1b": 'v',"E'1c": 'v',"E'1d": 'v',"E'1e": 'v',"E'2a": 'v',"E'2b": 'v',"E'2c": 'v',"E'2d": 'v',"E'2e": 'v',"A''1": 'h',"A''2": 'h',"E''1a":'h',"E''1b":'h',"E''1c":'h',"E''1d":'h',"E''1e":'h',"E''2a":'h',"E''2b":'h',"E''2c":'h',"E''2d":'h',"E''2e":'h'},
                            'D6h':{'A1g': 'v', 'A2g': 'v', 'B1g': 'h', 'B2g': 'h', 'A1u': 'h', 'A2u': 'h', 'B1u': 'v', 'B2u'   : 'v', 'E2g1': 'v', 'E2g2': 'v', 'E2g3'  : 'v', 'E1g1'  : 'h', 'E1g2'  : 'h', 'E1g3'  : 'h', 'E1g4'  : 'h', 'E1g5'  : 'h', 'E1g6'  : 'h', 'E1u1': 'v', 'E1u2': 'v', 'E1u3'  : 'v', 'E1u4'  : 'v', 'E1u5'  : 'v', 'E1u6'  : 'v', 'E2u1'  : 'h', 'E2u2'  : 'h', 'E2u3'  : 'h','E2u4'  : 'h', 'E2u5'  : 'h', 'E2u6'  : 'h'},
                            'D8h':{'A1g': 'v', 'A2g': 'v', 'B1g': 'v', 'B2g': 'v', 'A1u': 'h', 'A2u': 'h', 'B1u': 'h', 'B2u'   : 'h', 'E1g1': 'h', 'E1g2': 'h', 'E1g3'  : 'h', 'E1g4'  : 'h', 'E2g1': 'h', 'E2g2': 'h', 'E2g3'  : 'h', 'E2g4'  : 'h', 'E3g1': 'h', 'E3g2': 'h', 'E3g3'  : 'h', 'E3g4'  : 'h', 'E1u1': 'v','E1u2': 'v', 'E1u3': 'v','E1u4': 'v', 'E2u1': 'v','E2u2': 'v', 'E2u3': 'v','E2u4': 'v', 'E3u1': 'v','E3u2': 'v', 'E3u3': 'v','E3u4': 'v'}, 
                            'Td':{'A1'    :'v','A2'    :'v','Ez'    :'h','Ex'    :'h','Ey'    :'h`','T1(a)' :'v','T1(b)' :'v','T1(c)' :'v','T1(d)' :'v','T2(a)' :'h','T2(b)' :'h','T2(c)' :'h','T2(d)' :'h',},
                'Oh':{'A1g':'v','A2g':'v','Ega':'v','Egb':'v','Egc':'v','Egd':'v','T1gx':'v','T1gy':'v','T1gz':'v','T2ga':'v','T2gx':'v','T2gy':'v','T2gz':'v','A1u':'h','A2u':'h','Eua':'h','Eub':'h','Euc':'h','Eud':'h','T1ux':'h','T1uy':'h','T1uz':'h','T2ux1':'h','T2uy1':'h','T2uz1':'h','T2ux2':'h','T2uy2':'h','T2uz2':'h',},
                }

symm_typog_lookup = {'D4h':'D$_{4h}$','D4 ':'D$_{4 }$','D2 ':'D$_{2 }$','D2d':'D$_{2d}$','D2h':'D$_{2h}$','S4 ':'S$_{4 }$','C4 ':'C$_{4 }$','C4h':'C$_{4h}$','C1 ':'C$_{1 }$','Ci ':'C$_{i }$','C2h':'C$_{2h}$','Cs ':'C$_{s }$','C2 ':'C$_{2 }$','C2v':'C$_{2v}$','C4v':'C$_{4v}$',
                    'D4':'D$_{4 }$','D2':'D$_{2 }$','S4':'S$_{4 }$','C4':'C$_{4 }$','C1':'C$_{1 }$','Ci':'C$_{i }$','Cs':'C$_{s }$','C2':'C$_{2 }$',
                    'S3?':'S3?', 'D3':'D$_{3 }$','D3h':'D$_{3h}$', 'C3':'C$_{3 }$', 'C3v':'C$_{3v}$',
                    'D6h': 'D$_{6h}$','C6h': 'C$_{6h}$','D6' : 'D$_{6 }$' ,'C6v': 'C$_{6v}$','D3d': 'D$_{3d }$', 'S6':'S$_{6 }$',
                    'D5h': 'D$_{5h}$','C5h': 'C$_{5h}$','D5': 'D$_{5 }$','C5v': 'C$_{5v}$','C5': 'C$_{5 }$',
                    'C3h':'C$_{3h}$', 'C6':'C$_{6 }$',
                    'Td':'T$_{d}$','T' : 'T', 'Oh':'O$_{h}$', 'O':'O',
                    'D8h':'D$_{8h}$','S8':'S$_{8}$',
                    'unknown':'unknown', 'D4d': 'D$_{4d}$', 'D7h': 'D$_{7h}$',
                    'D7' : 'D$_{7 }$', 'C7h': 'C$_{7h}$','C7v': 'C$_{7v}$','C7' : 'C$_{7 }$', 'Th' : 'T$_{h }$',
                     'C8h': 'C$_{8h}$',
                     'C8v': 'C$_{8v}$','D8' : 'D$_{8 }$', 'C8' : 'C$_{8 }$', "S6?":"S6?", "???":"???" 
}

symm_typog_html  = {'D4h':'D<sub>4h</sub>','D4 ':'D<sub>4 </sub>','D2 ':'D<sub>2 </sub>','D2d':'D<sub>2d</sub>','D2h':'D<sub>2h</sub>','S4 ':'S<sub>4 </sub>','C4 ':'C<sub>4 </sub>','C4h':'C<sub>4h</sub>','C1 ':'C<sub>1 </sub>','Ci ':'C<sub>i </sub>','C2h':'C<sub>2h</sub>','Cs ':'C<sub>s </sub>','C2 ':'C<sub>2 </sub>','C2v':'C<sub>2v</sub>','C4v':'C<sub>4v</sub>',
                    'D4':'D<sub>4 </sub>','D2':'D<sub>2 </sub>','S4':'S<sub>4 </sub>','C4':'C<sub>4 </sub>','C1':'C<sub>1 </sub>','Ci':'C<sub>i </sub>','Cs':'C<sub>s </sub>','C2':'C<sub>2 </sub>',
                    'S3?':'S3?', 'D3':'D<sub>3 </sub>','C3':'C<sub>3 </sub>', 'C3v':'C<sub>3v</sub>',
                    'D6h': 'D<sub>6h</sub>','C6h': 'C<sub>6h</sub>','D3h': 'D<sub>3h</sub>','D6' : 'D<sub>6 </sub>' ,'C6v': 'C<sub>6v</sub>','D3d': 'D<sub>3d </sub>', 'S6':'S<sub>6 </sub>',
                    'D5h': 'D<sub>5h</sub>','C5h': 'C<sub>5h</sub>','D5': 'D<sub>5 </sub>','C5v': 'C<sub>5v</sub>','C5': 'C<sub>5 </sub>',
                    'C3h':'C<sub>3h</sub>', 'C6':'C<sub>6 </sub>',
                    'D4d':'D<sub>4d</sub>','S8':'S<sub>8</sub>',
                    'Td':'T<sub>d</sub>','T' : 'T','Oh':'O<sub>h</sub>','O':'O', "S6?":"S6?", "???":"???",
                    'unknown':'unknown','D7h' : 'D<sub>7h</sub>','D7'  : 'D<sub>7</sub>' ,'C7h' : 'C<sub>7h</sub>','C7v' : 'C<sub>7v</sub>','C7'  : 'C<sub>7</sub>' ,'Th'  : 'T<sub>h</sub>' ,'D8h' : 'D<sub>8h</sub>','C8h' : 'C<sub>8h</sub>','C8v' : 'C<sub>8v</sub>','D8'  : 'D<sub>8</sub>' ,'C8'  : 'C<sub>8</sub>' ,
                    }

nms = 'B$_{2g}$,B$_{1g}$,E$_{u}$x,E$_{u}$y,A$_{1g}$,A$_{2g}$,B$_{2u}$,B$_{1u}$,A$_{2u}$,E$_{g}$x,E$_{g}$y,A$_{1u}$'.split(',')

irrep_typog       = {'A1g':'A$_{1g}$','A2g':'A$_{2g}$','A1u':'A$_{1u}$','A2u':'A$_{2u}$','Egx':'E$_{g(x)}$','Egy':'E$_{g(y)}$','Egx+y':'E$_{g(x+y)}$','Egx-y':'E$_{g(x-y)}$','Eux':'E$_{u(x)}$','Euy':'E$_{u(y)}$','Eux+y':'E$_{u(x+y)}$','Eux-y':'E$_{u(x-y)}$',
                     'Ag':'A$_{g}$','B1g':'B$_{1g}$','B2g':'B$_{2g}$','B3g':'B$_{3g}$','Au':'A$_{u}$','B1u':'B$_{1u}$','B2u':'B$_{2u}$','B3u':'B$_{3u}$',#D2h
                     'A1':'A$_{1}$','A2':'A$_{2}$','B1':'B$_{1}$','B2':'B$_{2}$',#C2v
                     'Bg':'B$_{g}$', 'Bu':'B$_{u}$','Ex+y':'Ex+y','Ex-y':'Ex-y',
                     'Ex':'Ex','Ey':'Ey',
                     'Eg1':'E$_{g}$1','Eg2':'E$_{g}$2','Eg3':'E$_{g}$3','Eu1':'E$_{u}$1','Eu2':'E$_{u}$2','Eu3':'E$_{u}$3','Eu4':'E$_{u}$4','Eu5':'E$_{u}$5','Eu6':'E$_{u}$6',
                     "A'1"  : "A'1" , "A'2"  : "A'2" , "E'1"  : "E'1" , "E'2"  : "E'2" , "E'3"  : "E'3" , "A''1" : "A''1", "A''2" : "A''2", "E''1" : "E''1","E''2" : "E''2","E''3" : "E''3" ,
                     'E1g1':'E$_{1g}$1', 'E1g2':'E$_{1g}$2', 'E1g3':'E$_{1g}$3', 'E1g5':'E$_{1g}$5', 'E1g6':'E$_{1g}$6', 'E2g1':'E$_{2g}$1', 'E2g2':'E$_{2g}$2', 'E2g3':'E$_{2g}$3', 'E2g4':'E$_{2g}$4', 'E2g5':'E$_{2g}$5', 'E2g6':'E$_{2g}$6', 'E1u1':'E$_{1u}$1', 'E1u2':'E$_{1u}$2', 'E1u3':'E$_{1u}$3', 'E1u4':'E$_{1u}$4', 'E1u5':'E$_{1u}$5', 'E1u6':'E$_{1u}$6', 'E2u1':'E$_{2u}$1','E2u2':'E$_{2u}$2', 'E2u3':'E$_{2u}$3', 'E2u4':'E$_{2u}$4','E2u5':'E$_{2u}$5', 'E2u6':'E$_{2u}$6',
"E'1a":"E'1a","E'1b":"E'1b","E'1c":"E'1c","E'1d":"E'1d","E'1e":"E'1e","E'2a":"E'2a","E'2b":"E'2b","E'2c":"E'2c","E'2d":"E'2d","E'2e":"E'2e","E''1a":"E''1a","E''1b":"E''1b","E''1c":"E''1c","E''1d":"E''1d","E''1e":"E''1e","E''2a":"E''2a","E''2b":"E''2b","E''2c":"E''2c","E''2d":"E''2d","E''2e":"E''2e",                     
'E1a':'E$_{1}$a','E1b':'E$_{1}$b','E2a':'E$_{2}$a','E2b':'E$_{2}$b','E3a':'E$_{3}$a','E3b':'E$_{3}$b','E3c':'E$_{3}$c','E3d':'E$_{3}$d','E3e':'E$_{3}$e','E3f':'E$_{3}$f','E3g':'E$_{3}$g','E3h':'E$_{3}$h',
'Ez'   :'Ez','T1(a)':'T$_{1}$(a)','T1(b)':'T$_{1}$(b)','T1(c)':'T$_{1}$(c)','T1(d)':'T$_{1}$(d)','T2(a)':'T$_{2}$(a)','T2(b)':'T$_{2}$(b)','T2(c)':'T$_{2}$(c)','T2(d)':'T$_{2}$(d)',
'Egz':'E$_{g}$z','Euz':'E$_{u}$z',
"A1'"  : "A$_{1}$'" , "A1''" : "A$_{1}$''", "A2'"  : "A$_{2}$'" , "A2''" : "A$_{2}$''", "E'4"  : "E'4"  ,"E'5"  : "E'5"  ,"E'6"  : "E'6"  ,"E'7"  : "E'7"  ,"E''4" : "E''4" ,"E''5" : "E''5" ,"E''6" : "E''6" ,"E''7" : "E''7" ,"E1g4" : "E$_{1g}$4", "E3g1" : "E$_{3g}$1", 
"E3g2" : "E$_{3g}$2", "E3g3" : "E$_{3g}$3", "E3g4" : "E$_{3g}$4", "E3u1" : "E$_{3u}$1", "E3u2" : "E$_{3u}$2", "E3u3" : "E$_{3u}$3", "E3u4" : "E$_{3u}$4", "Ega"  : "E$_{g}$a" , "Egb"  : "E$_{g}$b" , "Egc"  : "E$_{g}$c" , "Egd"  : "E$_{g}$d" , "T1gx" : "T$_{1g}$x", "T1gy" : "T$_{1g}$y", "T1gz" : "T$_{1g}$z", "T2gx1": "T$_{2g}$x1","T2gy1": "T$_{2g}$y1","T2gz1": "T$_{2g}$z1","T2gx2": "T$_{2g}$x2","T2gy2": "T$_{2g}$y2","T2gz2": "T$_{2g}$z2",
"T1ux" : "T$_{1u}$x", "T1uy" : "T$_{1u}$y", "T1uz" : "T$_{1u}$z", "T2ux" : "T$_{2u}$x", "T2uy" : "T$_{2u}$y", "T2uz" : "T$_{2u}$z",  }

irrep_typog_html  = {'A1g':'A<sub>1g</sub>','A2g':'A<sub>2g</sub>','Egx':'E<sub>g</sub>x','Egy':'E<sub>g</sub>y','Egx+y':'E<sub>g</sub>x+y','Egx-y':'E<sub>g</sub>x-y','Eux':'E<sub>u</sub>x','Euy':'E<sub>u</sub>y','Eux+y':'E<sub>u</sub>x+y','Eux-y':'E<sub>u</sub>x-y',
                     'A1u':'A<sub>1u</sub>','A2u':'A<sub>2u</sub>','Ex':'Ex','Ey':'Ey','Ex+y':'Ex+y','Ex-y':'Ex-y',
                     'Ag':'A<sub>g</sub>','B1g':'B<sub>1g</sub>','B2g':'B<sub>2g</sub>','B3g':'B<sub>3g</sub>','Au':'A<sub>u</sub>','B1u':'B<sub>1u</sub>','B2u':'B<sub>2u</sub>','B3u':'B<sub>3u</sub>',#D2h
                     'A1':'A<sub>1</sub>','A2':'A<sub>2</sub>','B1':'B<sub>1</sub>','B2':'B<sub>2</sub>',#C2v
                     'Bg':'B<sub>g</sub>', 'Bu':'B<sub>u</sub>', 'E1':'E<sub>1</sub>','E2':'E<sub>2</sub>','E3':'E<sub>3</sub>',
                     "A'1"  : "A'1" , "A'2"  : "A'2" , "E'1"  : "E'1" , "E'2"  : "E'2" , "E'3"  : "E'3" , "A''1" : "A''1", "A''2" : "A''2", "E''1" : "E''1","E''2" : "E''2","E''3" : "E''3",
                      'Eg1':'E<sub>g</sub>1','Eg2':'E<sub>g</sub>2','Eg3':'E<sub>g</sub>3','Eu1':'E<sub>u</sub>1','Eu2':'E<sub>u</sub>2','Eu3':'E<sub>u</sub>3','Eu4':'E<sub>u</sub>4','Eu5':'E<sub>u</sub>5','Eu6':'E<sub>u</sub>6',
                     'E1g1':'E<sub>1g</sub>1', 'E1g2':'E<sub>1g</sub>2', 'E1g3':'E<sub>1g</sub>3', 'E1g5':'E<sub>1g</sub>5', 'E1g6':'E<sub>1g</sub>6', 'E2g1':'E<sub>2g</sub>1', 'E2g2':'E<sub>2g</sub>2', 'E2g3':'E<sub>2g</sub>3', 'E2g4':'E<sub>2g</sub>4', 'E2g5':'E<sub>2g</sub>5', 'E2g6':'E<sub>2g</sub>6', 'E1u1':'E<sub>1u</sub>1', 'E1u2':'E<sub>1u</sub>2', 'E1u3':'E<sub>1u</sub>3', 'E1u4':'E<sub>1u</sub>4', 'E1u5':'E<sub>1u</sub>5', 'E1u6':'E<sub>1u</sub>6', 'E2u1':'E<sub>2u</sub>1','E2u2':'E<sub>2u</sub>2', 'E2u3':'E<sub>2u</sub>3', 'E2u4':'E<sub>2u</sub>4','E2u5':'E<sub>2u</sub>5', 'E2u6':'E<sub>2u</sub>6',
                    "E'1a":"E'1a","E'1b":"E'1b","E'1c":"E'1c","E'1d":"E'1d","E'1e":"E'1e","E'2a":"E'2a","E'2b":"E'2b","E'2c":"E'2c","E'2d":"E'2d","E'2e":"E'2e","E''1a":"E''1a","E''1b":"E''1b","E''1c":"E''1c","E''1d":"E''1d","E''1e":"E''1e","E''2a":"E''2a","E''2b":"E''2b","E''2c":"E''2c","E''2d":"E''2d","E''2e":"E''2e",
                    'E1a':'E<sub>1</sub>a', 'E1b':'E<sub>1</sub>b', 'E2a':'E<sub>2</sub>a', 'E2b':'E<sub>2</sub>b', 'E3a':'E<sub>3</sub>a', 'E3b':'E<sub>3</sub>b', 'E3c':'E<sub>3</sub>c', 'E3d':'E<sub>3</sub>d', 'E3e':'E<sub>3</sub>e', 'E3f':'E<sub>3</sub>f', 'E3g':'E<sub>3</sub>g', 'E3h':'E<sub>3</sub>h',
                    'Eg':'E<sub>g</sub>','Eu':'E<sub>u</sub>','Egz':'E<sub>g</sub>z','Euz':'E<sub>u</sub>z',
                    'Ez'   :'Ez','T1(a)':'T<sub>1</sub>a','T1(b)':'T<sub>1</sub>b','T1(c)':'T<sub>1</sub>c','T1(d)':'T<sub>1</sub>d','T2(a)':'T<sub>2</sub>a','T2(b)':'T<sub>2</sub>b','T2(c)':'T<sub>2</sub>c','T2(d)':'T<sub>2</sub>d',
'T1g(a)':'T<sub>1g</sub>a','T1g(b)':'T<sub>1g</sub>b','T1g(c)':'T<sub>1g</sub>c','T1g(d)':'T<sub>1g</sub>d','T2g(a)':'T<sub>2g</sub>a','T2g(b)':'T<sub>2g</sub>b','T2g(c)':'T<sub>2g</sub>c','T2g(d)':'T<sub>2g</sub>d',
'T1u(a)':'T<sub>1u</sub>a','T1u(b)':'T<sub>1u</sub>b','T1u(c)':'T<sub>1u</sub>c','T1u(d)':'T<sub>1u</sub>d','T2u(a)':'T<sub>2u</sub>a','T2u(b)':'T<sub>2u</sub>b','T2u(c)':'T<sub>2u</sub>c','T2u(d)':'T<sub>2u</sub>d',
"A1'"  : "A<sub>1</sub>'" , "A1''" : "A<sub>1</sub>''", "A2'"  : "A<sub>2</sub>'" , "A2''" : "A<sub>2</sub>''", "E'4"  : "E'4"  ,"E'5"  : "E'5"  ,"E'6"  : "E'6"  ,"E'7"  : "E'7"  ,"E''4" : "E''4" ,"E''5" : "E''5" ,"E''6" : "E''6" ,"E''7" : "E''7" ,"E1g4" : "E<sub>1g</sub>4", "E3g1" : "E<sub>3g</sub>1", "E3g2" : "E<sub>3g</sub>2", "E3g3" : "E<sub>3g</sub>3", "E3g4" : "E<sub>3g</sub>4", "E3u1" : "E<sub>3u</sub>1", "E3u2" : "E<sub>3u</sub>2", "E3u3" : "E<sub>3u</sub>3", "E3u4" : "E<sub>3u</sub>4", "Ega"  : "E<sub>g</sub>a" , "Egb"  : "E<sub>g</sub>b" , "Egc"  : "E<sub>g</sub>c" , "Egd"  : "E<sub>g</sub>d" , "T1gx" : "T<sub>1g</sub>x", "T1gy" : "T<sub>1g</sub>y", "T1gz" : "T<sub>1g</sub>z", "T2gx1": "T<sub>2g</sub>x1","T2gy1": "T<sub>2g</sub>y1","T2gz1": "T<sub>2g</sub>z1","T2gx2": "T<sub>2g</sub>x2","T2gy2": "T<sub>2g</sub>y2","T2gz2": "T<sub>2g</sub>z2","T1ux" : "T<sub>1u</sub>x", "T1uy" : "T<sub>1u</sub>y", "T1uz" : "T<sub>1u</sub>z", "T2ux" : "T<sub>2u</sub>x", "T2uy" : "T<sub>2u</sub>y", "T2uz" : "T<sub>2u</sub>z",  
'E1g':'E<sub>1g</sub>','E1u':'E<sub>1u</sub>','E2g':'E<sub>2g</sub>','E2u':'E<sub>2u</sub>',}

mondrian_transform_dict = {'D2h':{'[1, 1, 1, 1, 1, 1, 1, 1]':'No Change','[1, 1, 0, 0, 1, 1, 0, 0]':'Ag -> Ag, B1g -> Ag, B2g -> Bg, B3g -> Bg, Au -> Au, B1u -> Au, B2u -> Bu, B3u -> Bu', '[1, 0, 1, 0, 1, 0, 1, 0]':'Ag -> Ag, B1g -> Bg, B2g -> Ag, B3g -> Bg, Au -> Au, B1u -> Bu, B2u -> Au, B3u -> Bu','[1, 0, 0, 1, 1, 0, 0, 1]':'Ag -> Ag, B1g -> Bg, B2g -> Bg, B3g -> Ag, Au -> Au, B1u -> Bu, B2u -> Bu, B3u -> Au','[1, 1, 1, 1, 0, 0, 0, 0]':'Ag -> A, B1g -> B1, B2g -> B2, B3g -> B3, Au -> A, B1u -> B1, B2u -> B2, B3u -> B3', '[1, 1, 0, 0, 0, 0, 1, 1]':'Ag -> A1, B1g -> A1, B2g -> B1, B3g -> B1, Au -> A2, B1u -> A2, B2u -> B2, B3u -> B2','[1, 0, 1, 0, 0, 1, 0, 1]':'Ag -> A1, B1g -> B1, B2g -> A1, B3g -> B1, Au -> A2, B1u -> B2, B2u -> A2, B3u -> B2','[1, 0, 0, 1, 0, 1, 1, 0]':'Ag -> A1, B1g -> B1, B2g -> B1, B3g -> A1, Au -> A2, B1u -> B2, B2u -> B2, B3u -> A2','[1, 1, 0, 0, 0, 0, 0, 0]':'Ag -> A, B1g -> A, B2g -> B, B3g -> B, Au -> A, B1u -> A, B2u -> B, B3u -> B', '[1, 0, 1, 0, 0, 0, 0, 0]':'Ag -> A, B1g -> B, B2g -> A, B3g -> B, Au -> A, B1u -> B, B2u -> A, B3u -> B', '[1, 0, 0, 1, 0, 0, 0, 0]':'Ag -> A, B1g -> B, B2g -> B, B3g -> A, Au -> A, B1u -> B, B2u -> B, B3u -> A', '[1, 0, 0, 0, 1, 0, 0, 0]':'Ag -> Ag, B1g -> Ag, B2g -> Ag, B3g -> Ag, Au -> Au, B1u -> Au, B2u -> Au, B3u -> Au', '[1, 0, 0, 0, 0, 1, 0, 0]':"Ag -> A', B1g -> A', B2g -> A'', B3g -> A'', Au -> A'', B1u -> A'', B2u -> A', B3u -> A'", '[1, 0, 0, 0, 0, 0, 1, 0]':"Ag -> A', B1g -> A'', B2g -> A', B3g -> A'', Au -> A'', B1u -> A', B2u -> A'', B3u -> A'", '[1, 0, 0, 0, 0, 0, 0, 1]':"Ag -> A', B1g -> A'', B2g -> A'', B3g -> A', Au -> A'', B1u -> A', B2u -> A', B3u -> A''", '[1, 0, 0, 0, 0, 0, 0, 0]':'All -> A'},
'D4h':{'[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]':'No Change',
       '[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]':
       'A1g -> Ag, A2g -> Ag, B1g -> Bg, B2g -> Bg, Eg -> Eg, A1u -> Au, A2u -> Au, B1u -> Bu, B2u -> Bu, Eu -> Eu',
       '[1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0]':
       'A1g -> Ag, A2g -> B1g, B1g -> Ag, B2g -> B1g, Eg -> B2g + B3g, A1u -> Au, A2u -> B1u, B1u -> Au, B2u -> B1u, Eu -> B2u + B3u',
       '[1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1]':
       'A1g -> Ag, A2g -> Ag, B1g -> Bg, B2g -> Bg, Eg -> Eg, A1u -> Au, A2u -> Au, B1u -> Bu, B2u -> Bu, Eu -> Eu',
       '[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]':'D4 ',
       '[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]':'C4v',
       '[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0]':'D2d',
       '[1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1]':'D2d',
       '[1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]':'C2h',
       '[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]':'C2h',
       '[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]':'C2h',
       '[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]':'C2h',
       '[1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]':'C2h',         
       '[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]':'C2v',
       '[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]':'C2v',
       '[1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]':'C2v',
       '[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]':'C2v',
       '[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]':'C2v',
       '[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]':'C2v',
       '[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]':'D2 ',
       '[1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'D2 ',
       '[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]':'S4 ',
       '[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C4 ',
       '[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2 ',
       '[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2 ',
       '[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C2 ',
       '[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]':'C2 ',
       '[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]':'C2 ',
       '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]':'Ci ',
       '[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]':'Cs ',
       '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]':'Cs ',
       '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]':'Cs ',
       '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]':'Cs ',
       '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]':'Cs ',
       '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]':'C1 ',},
                       
                        }

# Checklist for new symmetry:
# indicate ordered ops
# expanded point group table
# mondrian lookup table (remember C1)
#