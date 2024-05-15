from .scsd import scsd_model
from numpy import array
model_objs_dict = {}

model_objs_dict['texaphyrin_2'] = scsd_model('texaphyrin_2', array([['-1.0889', '0.0', '-3.462', 'C'], ['-0.6722', '0.0', '-4.8482', 'C'], ['0.6722', '0.0', '-4.8482', 'C'], ['1.0889', '0.0', '-3.462', 'C'], ['2.4346', '0.0', '-3.0759', 'C'], ['3.0123', '0.0', '-1.8193', 'C'], ['3.1467', '-0.0', '0.353', 'C'], ['4.4991', '0.0', '-0.1281', 'C'], ['4.4173', '0.0', '-1.482', 'C'], ['2.6445', '-0.0', '1.6907', 'C'], ['0.7125', '-0.0', '3.0875', 'C'], ['1.3983', '-0.0', '4.317', 'C'], ['0.7189', '-0.0', '5.499', 'C'], ['-0.7189', '-0.0', '5.499', 'C'], ['-1.3983', '-0.0', '4.317', 'C'], ['-0.7125', '-0.0', '3.0875', 'C'], ['-2.6445', '-0.0', '1.6907', 'C'], ['-3.1467', '-0.0', '0.353', 'C'], ['-3.0123', '0.0', '-1.8193', 'C'], ['-4.4173', '0.0', '-1.482', 'C'], ['-4.4991', '0.0', '-0.1281', 'C'], ['-2.4346', '0.0', '-3.0759', 'C'], ['0.0', '0.0', '-2.6288', 'N'], ['2.2737', '0.0', '-0.6722', 'N'], ['1.3472', '-0.0', '1.8549', 'N'], ['-1.3472', '-0.0', '1.8549', 'N'], ['-2.2737', '0.0', '-0.6722', 'N']]), 'C2v' , pca = {"A1":array([[-0.01008, 0.0, 0.00106, -0.01247, -0.0, 0.22857, 0.01247, -0.0, 0.22857, 0.01008, 0.0, 0.00106, 0.01986, 0.0, -0.03304, 0.02474, 0.0, -0.01356, 0.00036, -0.0, -0.00197, -0.02322, 0.0, -0.00297, -0.00862, -0.0, -0.00626, -0.01378, 0.0, -0.01772, -0.00117, -0.0, 0.01453, -0.00278, 0.0, 0.00406, -0.00613, -0.0, 0.00601, 0.00613, -0.0, 0.00601, 0.00278, 0.0, 0.00406, 0.00117, -0.0, 0.01453, 0.01378, 0.0, -0.01772, -0.00036, -0.0, -0.00197, -0.02474, 0.0, -0.01356, 0.00862, -0.0, -0.00626, 0.02322, 0.0, -0.00297, -0.01986, 0.0, -0.03304, 0.0, 0.0, -0.13206, 0.02946, 0.0, -0.01992, -0.00311, -0.0, 0.00515, 0.00311, -0.0, 0.00515, -0.02946, 0.0, -0.01992], [0.00335, 0.0, -0.03518, -0.00394, 0.0, -0.01398, 0.00394, 0.0, -0.01398, -0.00335, 0.0, -0.03518, -0.01042, 0.0, -0.02017, -0.04721, 0.0, -0.00558, -0.06, 0.0, -0.00858, -0.05685, -0.0, -0.00446, -0.04796, -0.0, -0.00074, -0.03227, -0.0, -0.0043, 0.0004, -0.0, 0.01984, 0.00116, -0.0, 0.0241, 0.00213, -0.0, 0.02225, -0.00213, -0.0, 0.02225, -0.00116, -0.0, 0.0241, -0.0004, -0.0, 0.01984, 0.03227, -0.0, -0.0043, 0.06, 0.0, -0.00858, 0.04721, 0.0, -0.00558, 0.04796, -0.0, -0.00074, 0.05685, -0.0, -0.00446, 0.01042, 0.0, -0.02017, 0.0, 0.0, -0.04083, -0.0654, 0.0, -0.00793, -0.02508, -0.0, 0.01479, 0.02508, -0.0, 0.01479, 0.0654, 0.0, -0.00793]]), "A2":array([[-0.0, 0.04203, -0.0, -0.0, 0.03193, -0.0, 0.0, -0.03193, -0.0, -0.0, -0.04203, -0.0, -0.0, -0.07404, -0.0, -0.0, -0.04024, -0.0, -0.0, 0.03991, 0.0, -0.0, 0.01536, -0.0, -0.0, -0.0422, -0.0, -0.0, 0.06095, 0.0, -0.0, 0.02556, -0.0, -0.0, 0.05113, -0.0, 0.0, 0.02436, -0.0, -0.0, -0.02436, -0.0, -0.0, -0.05113, -0.0, -0.0, -0.02556, -0.0, -0.0, -0.06095, -0.0, -0.0, -0.03991, -0.0, -0.0, 0.04024, -0.0, -0.0, 0.0422, -0.0, -0.0, -0.01536, 0.0, -0.0, 0.07404, 0.0, 0.0, 0.0, -0.0, -0.0, 0.01165, -0.0, -0.0, 0.04063, -0.0, -0.0, -0.04063, 0.0, -0.0, -0.01165, 0.0], [-0.0, 0.03237, -0.0, 0.0, 0.02774, -0.0, 0.0, -0.02774, -0.0, -0.0, -0.03237, -0.0, -0.0, -0.05021, -0.0, -0.0, -0.0238, -0.0, -0.0, 0.01653, 0.0, -0.0, 0.07056, -0.0, -0.0, 0.04559, -0.0, -0.0, 0.01214, 0.0, -0.0, -0.02564, -0.0, -0.0, -0.08058, -0.0, -0.0, -0.04646, -0.0, -0.0, 0.04646, -0.0, -0.0, 0.08058, -0.0, -0.0, 0.02564, -0.0, -0.0, -0.01214, -0.0, -0.0, -0.01653, -0.0, -0.0, 0.0238, -0.0, -0.0, -0.04559, -0.0, -0.0, -0.07056, 0.0, -0.0, 0.05021, 0.0, 0.0, 0.0, -0.0, -0.0, -0.04153, 0.0, -0.0, -0.02684, -0.0, -0.0, 0.02684, 0.0, -0.0, 0.04153, -0.0]]), "B1":array([[0.041, 0.0, 0.02195, 0.00429, -0.0, 0.02237, 0.00429, 0.0, -0.02237, 0.041, 0.0, -0.02195, 0.04865, -0.0, -0.05194, 0.03489, -0.0, -0.02518, -0.00854, 0.0, -0.02096, 0.01234, -0.0, 0.01638, 0.04451, -0.0, 0.01369, -0.00918, 0.0, -0.04005, -0.02972, 0.0, 0.00578, -0.00645, -0.0, -0.02544, 0.01814, -0.0, -0.01777, 0.01814, -0.0, 0.01777, -0.00645, -0.0, 0.02544, -0.02972, 0.0, -0.00578, -0.00918, 0.0, 0.04005, -0.00854, 0.0, 0.02096, 0.03489, 0.0, 0.02518, 0.04451, 0.0, -0.01369, 0.01234, -0.0, -0.01638, 0.04865, -0.0, 0.05194, 0.07037, 0.0, -0.0, 0.00741, -0.0, -0.0481, -0.01995, 0.0, -0.00047, -0.01995, 0.0, 0.00047, 0.00741, 0.0, 0.0481], [0.01297, 0.0, 0.00673, 0.01571, 0.0, -0.00102, 0.01571, -0.0, 0.00102, 0.01297, 0.0, -0.00673, -0.00398, -0.0, 0.03024, -0.01517, 0.0, 0.03201, -0.02385, 0.0, 0.01612, -0.0325, 0.0, 0.03724, -0.02807, -0.0, 0.04361, -0.04326, -0.0, 0.00143, -0.01554, 0.0, -0.01906, 0.02202, 0.0, -0.03271, 0.07824, -0.0, -0.03735, 0.07824, 0.0, 0.03735, 0.02202, 0.0, 0.03271, -0.01554, 0.0, 0.01906, -0.04326, 0.0, -0.00143, -0.02385, 0.0, -0.01612, -0.01517, -0.0, -0.03201, -0.02807, -0.0, -0.04361, -0.0325, 0.0, -0.03724, -0.00398, 0.0, -0.03024, -0.01037, -0.0, 0.0, -0.01348, -0.0, 0.02047, -0.04658, 0.0, -0.01799, -0.04658, 0.0, 0.01799, -0.01348, 0.0, -0.02047]]), "B2":array([[0.0, 0.0207, -0.0, 0.0, 0.18117, -0.0, 0.0, 0.18117, 0.0, 0.0, 0.0207, 0.0, 0.0, -0.04539, -0.0, 0.0, -0.02159, -0.0, 0.0, -0.012, 0.0, 0.0, 0.02234, 0.0, 0.0, 0.01661, -0.0, 0.0, -0.00704, -0.0, 0.0, -0.01286, 0.0, 0.0, 0.01076, 0.0, 0.0, 0.03989, 0.0, -0.0, 0.03989, 0.0, -0.0, 0.01076, 0.0, 0.0, -0.01286, 0.0, 0.0, -0.00704, 0.0, 0.0, -0.012, -0.0, 0.0, -0.02159, -0.0, 0.0, 0.01661, -0.0, 0.0, 0.02234, -0.0, 0.0, -0.04539, -0.0, 0.0, -0.07674, 0.0, 0.0, -0.03886, 0.0, 0.0, -0.03242, 0.0, -0.0, -0.03242, -0.0, 0.0, -0.03886, 0.0], [-0.0, -0.04303, -0.0, -0.0, -0.02868, 0.0, 0.0, -0.02868, 0.0, -0.0, -0.04303, 0.0, 0.0, -0.037, -0.0, 0.0, 0.00561, 0.0, 0.0, 0.04721, 0.0, 0.0, 0.10821, 0.0, 0.0, 0.08109, 0.0, 0.0, 0.04612, -0.0, 0.0, -0.01206, 0.0, -0.0, -0.02044, 0.0, 0.0, -0.026, 0.0, -0.0, -0.026, 0.0, 0.0, -0.02044, 0.0, 0.0, -0.01206, 0.0, 0.0, 0.04612, 0.0, 0.0, 0.04721, -0.0, 0.0, 0.00561, 0.0, 0.0, 0.08109, 0.0, 0.0, 0.10821, 0.0, 0.0, -0.037, -0.0, -0.0, -0.05586, 0.0, 0.0, -0.01146, -0.0, -0.0, -0.00515, -0.0, 0.0, -0.00515, -0.0, 0.0, -0.01146, -0.0]])}, database_path = 'texaphyrin_2_scsd_20231019.pkl', maxdist = 1.75, mondrian_limits = [-1.0, -1.0], smarts = '[N,n]1~[C,c]2~[C,c]~[C,c]~[C,c]~1[C,c]~[C,c]1~[C,c]~[C,c]~[C,c](~[N,n]~1)~[C,c]~[N,n]~[C,c]1~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1[N,n]~[C,c]~[C,c]1~[N,n]~[C,c](~[C,c]~2)~[C,c]~[C,c]~1')
model_objs_dict['porphycene'] = scsd_model('porphycene', array([['0.7052', '2.415', '-0.0', 'C'], ['1.5918', '3.5337', '-0.0', 'C'], ['2.8513', '3.0389', '-0.0', 'C'], ['2.7636', '1.6084', '-0.0', 'C'], ['3.8568', '0.7029', '-0.0', 'C'], ['3.8568', '-0.7029', '0.0', 'C'], ['2.7636', '-1.6084', '0.0', 'C'], ['2.8513', '-3.0389', '0.0', 'C'], ['1.5918', '-3.5337', '0.0', 'C'], ['0.7052', '-2.415', '0.0', 'C'], ['-0.7052', '-2.415', '0.0', 'C'], ['-1.5918', '-3.5337', '0.0', 'C'], ['-2.8513', '-3.0389', '0.0', 'C'], ['-2.7636', '-1.6084', '0.0', 'C'], ['-3.8568', '-0.7029', '0.0', 'C'], ['-3.8568', '0.7029', '-0.0', 'C'], ['-2.7636', '1.6084', '-0.0', 'C'], ['-2.8513', '3.0389', '-0.0', 'C'], ['-1.5918', '3.5337', '-0.0', 'C'], ['-0.7052', '2.415', '-0.0', 'C'], ['1.4336', '1.273', '-0.0', 'N'], ['1.4336', '-1.273', '0.0', 'N'], ['-1.4336', '-1.273', '0.0', 'N'], ['-1.4336', '1.273', '-0.0', 'N']]), 'D2h' , pca = {"Ag":array([[-0.00167, -0.05082, -0.0, -0.03617, -0.05969, -0.0, -0.02416, -0.03632, -0.0, 0.01526, -0.01426, 0.0, 0.02968, -0.00127, 0.0, 0.02968, 0.00127, 0.0, 0.01526, 0.01426, 0.0, -0.02416, 0.03632, 0.0, -0.03617, 0.05969, 0.0, -0.00167, 0.05082, 0.0, 0.00167, 0.05082, 0.0, 0.03617, 0.05969, 0.0, 0.02416, 0.03632, 0.0, -0.01526, 0.01426, -0.0, -0.02968, 0.00127, -0.0, -0.02968, -0.00127, -0.0, -0.01526, -0.01426, -0.0, 0.02416, -0.03632, -0.0, 0.03617, -0.05969, -0.0, 0.00167, -0.05082, -0.0, 0.02785, -0.02144, 0.0, 0.02785, 0.02144, 0.0, -0.02785, 0.02144, -0.0, -0.02785, -0.02144, -0.0], [-0.00333, 0.04021, -0.0, -0.00188, -0.03297, -0.0, -0.01406, -0.06443, -0.0, -0.02476, -0.00762, -0.0, -0.03057, -0.00807, -0.0, -0.03057, 0.00807, -0.0, -0.02476, 0.00762, -0.0, -0.01406, 0.06443, 0.0, -0.00188, 0.03297, 0.0, -0.00333, -0.04021, -0.0, 0.00333, -0.04021, -0.0, 0.00188, 0.03297, 0.0, 0.01406, 0.06443, 0.0, 0.02476, 0.00762, 0.0, 0.03057, 0.00807, 0.0, 0.03057, -0.00807, 0.0, 0.02476, -0.00762, 0.0, 0.01406, -0.06443, -0.0, 0.00188, -0.03297, -0.0, 0.00333, 0.04021, 0.0, -0.01498, 0.05101, 0.0, -0.01498, -0.05101, -0.0, 0.01498, -0.05101, -0.0, 0.01498, 0.05101, 0.0]]), "B1g":array([[-0.01688, -0.0161, -0.0, -0.02262, -0.07128, -0.0, -0.02828, -0.07755, -0.0, -0.0241, -0.02674, -0.0, -0.01145, -0.00708, -0.0, 0.01145, -0.00708, 0.0, 0.0241, -0.02674, 0.0, 0.02828, -0.07755, 0.0, 0.02262, -0.07128, 0.0, 0.01688, -0.0161, 0.0, 0.01688, 0.0161, 0.0, 0.02262, 0.07128, 0.0, 0.02828, 0.07755, 0.0, 0.0241, 0.02674, 0.0, 0.01145, 0.00708, 0.0, -0.01145, 0.00708, -0.0, -0.0241, 0.02674, -0.0, -0.02828, 0.07755, -0.0, -0.02262, 0.07128, -0.0, -0.01688, 0.0161, -0.0, -0.0196, 0.00337, 0.0, 0.0196, 0.00337, -0.0, 0.0196, -0.00337, -0.0, -0.0196, -0.00337, 0.0], [-0.01757, -0.01028, -0.0, -0.00159, 0.03306, 0.0, -0.01493, 0.02247, 0.0, -0.04796, -0.02615, 0.0, -0.02575, -0.01746, -0.0, 0.02575, -0.01746, -0.0, 0.04796, -0.02615, -0.0, 0.01493, 0.02247, -0.0, 0.00159, 0.03306, -0.0, 0.01757, -0.01028, -0.0, 0.01757, 0.01028, -0.0, 0.00159, -0.03306, -0.0, 0.01493, -0.02247, -0.0, 0.04796, 0.02615, -0.0, 0.02575, 0.01746, 0.0, -0.02575, 0.01746, 0.0, -0.04796, 0.02615, 0.0, -0.01493, -0.02247, 0.0, -0.00159, -0.03306, 0.0, -0.01757, 0.01028, 0.0, -0.05681, -0.06164, -0.0, 0.05681, -0.06164, -0.0, 0.05681, 0.06164, 0.0, -0.05681, 0.06164, 0.0]]), "B2g":array([[-0.0, -0.0, -0.06996, -0.0, 0.0, 0.007, 0.0, -0.0, 0.06812, 0.0, -0.0, 0.00912, 0.0, 0.0, 0.02005, 0.0, -0.0, -0.02005, 0.0, 0.0, -0.00912, -0.0, 0.0, -0.06812, -0.0, 0.0, -0.007, -0.0, 0.0, 0.06996, 0.0, 0.0, 0.06996, 0.0, 0.0, -0.007, -0.0, 0.0, -0.06812, 0.0, 0.0, -0.00912, -0.0, -0.0, -0.02005, -0.0, 0.0, 0.02005, 0.0, -0.0, 0.00912, 0.0, -0.0, 0.06812, 0.0, 0.0, 0.007, 0.0, -0.0, -0.06996, -0.0, 0.0, -0.07576, 0.0, -0.0, 0.07576, 0.0, -0.0, 0.07576, -0.0, 0.0, -0.07576], [-0.0, 0.0, -0.03785, 0.0, -0.0, -0.08074, -0.0, 0.0, -0.01118, -0.0, -0.0, 0.04408, 0.0, 0.0, 0.04223, 0.0, -0.0, -0.04223, -0.0, 0.0, -0.04408, -0.0, -0.0, 0.01118, 0.0, -0.0, 0.08074, 0.0, -0.0, 0.03785, 0.0, -0.0, 0.03785, -0.0, -0.0, 0.08074, 0.0, -0.0, 0.01118, -0.0, 0.0, -0.04408, -0.0, -0.0, -0.04223, -0.0, 0.0, 0.04223, -0.0, -0.0, 0.04408, 0.0, 0.0, -0.01118, -0.0, -0.0, -0.08074, -0.0, 0.0, -0.03785, -0.0, 0.0, 0.03392, 0.0, -0.0, -0.03392, 0.0, -0.0, -0.03392, -0.0, 0.0, 0.03392]]), "B3g":array([[0.0, -0.0, -0.00987, -0.0, -0.0, -0.08296, 0.0, -0.0, -0.06434, -0.0, -0.0, 0.01054, 0.0, -0.0, 0.04424, 0.0, 0.0, 0.04424, -0.0, 0.0, 0.01054, 0.0, 0.0, -0.06434, -0.0, -0.0, -0.08296, 0.0, -0.0, -0.00987, -0.0, -0.0, 0.00987, 0.0, -0.0, 0.08296, -0.0, 0.0, 0.06434, -0.0, 0.0, -0.01054, -0.0, 0.0, -0.04424, -0.0, -0.0, -0.04424, -0.0, -0.0, -0.01054, -0.0, -0.0, 0.06434, 0.0, -0.0, 0.08296, -0.0, 0.0, 0.00987, 0.0, -0.0, 0.03806, 0.0, 0.0, 0.03806, -0.0, 0.0, -0.03806, -0.0, -0.0, -0.03806], [0.0, 0.0, 0.01104, -0.0, 0.0, -0.01627, 0.0, -0.0, -0.0467, -0.0, 0.0, -0.03189, -0.0, 0.0, -0.11414, -0.0, -0.0, -0.11414, -0.0, -0.0, -0.03189, 0.0, 0.0, -0.0467, 0.0, -0.0, -0.01627, -0.0, -0.0, 0.01104, -0.0, -0.0, -0.01104, -0.0, -0.0, 0.01627, -0.0, 0.0, 0.0467, -0.0, -0.0, 0.03189, 0.0, -0.0, 0.11414, 0.0, 0.0, 0.11414, -0.0, 0.0, 0.03189, -0.0, -0.0, 0.0467, -0.0, -0.0, 0.01627, 0.0, 0.0, -0.01104, -0.0, -0.0, 0.02997, -0.0, 0.0, 0.02997, 0.0, 0.0, -0.02997, 0.0, -0.0, -0.02997]]), "Au":array([[-0.0, 0.0, 0.0172, -0.0, -0.0, 0.08061, -0.0, 0.0, 0.09123, 0.0, -0.0, 0.03437, -0.0, 0.0, 0.01774, -0.0, 0.0, -0.01774, -0.0, -0.0, -0.03437, 0.0, 0.0, -0.09123, 0.0, -0.0, -0.08061, 0.0, -0.0, -0.0172, -0.0, -0.0, 0.0172, -0.0, -0.0, 0.08061, -0.0, 0.0, 0.09123, -0.0, 0.0, 0.03437, -0.0, 0.0, 0.01774, -0.0, 0.0, -0.01774, -0.0, 0.0, -0.03437, -0.0, 0.0, -0.09123, 0.0, -0.0, -0.08061, -0.0, -0.0, -0.0172, -0.0, 0.0, -0.00885, -0.0, 0.0, 0.00885, -0.0, -0.0, -0.00885, -0.0, -0.0, 0.00885], [0.0, 0.0, 0.00866, 0.0, -0.0, -0.03509, 0.0, -0.0, 0.00348, 0.0, 0.0, 0.06465, -0.0, 0.0, 0.05123, -0.0, 0.0, -0.05123, -0.0, 0.0, -0.06465, -0.0, -0.0, -0.00348, -0.0, -0.0, 0.03509, 0.0, -0.0, -0.00866, 0.0, -0.0, 0.00866, 0.0, -0.0, -0.03509, 0.0, 0.0, 0.00348, -0.0, -0.0, 0.06465, -0.0, 0.0, 0.05123, -0.0, 0.0, -0.05123, -0.0, -0.0, -0.06465, -0.0, 0.0, -0.00348, -0.0, -0.0, 0.03509, -0.0, -0.0, -0.00866, -0.0, -0.0, 0.08688, -0.0, -0.0, -0.08688, -0.0, 0.0, 0.08688, -0.0, 0.0, -0.08688]]), "B1u":array([[-0.0, 0.0, -0.01503, 0.0, -0.0, -0.07396, -0.0, 0.0, -0.04742, -0.0, 0.0, 0.02205, -0.0, -0.0, 0.0484, -0.0, -0.0, 0.0484, -0.0, -0.0, 0.02205, -0.0, -0.0, -0.04742, 0.0, -0.0, -0.07396, -0.0, -0.0, -0.01503, 0.0, -0.0, -0.01503, 0.0, -0.0, -0.07396, 0.0, -0.0, -0.04742, -0.0, -0.0, 0.02205, -0.0, -0.0, 0.0484, -0.0, -0.0, 0.0484, -0.0, 0.0, 0.02205, -0.0, 0.0, -0.04742, -0.0, -0.0, -0.07396, 0.0, -0.0, -0.01503, -0.0, -0.0, 0.04314, -0.0, -0.0, 0.04314, -0.0, -0.0, 0.04314, -0.0, -0.0, 0.04314], [0.0, -0.0, -0.06006, -0.0, -0.0, -0.00337, 0.0, 0.0, 0.04794, -0.0, -0.0, 0.0176, -0.0, -0.0, 0.06505, -0.0, -0.0, 0.06505, -0.0, 0.0, 0.0176, -0.0, -0.0, 0.04794, 0.0, -0.0, -0.00337, 0.0, -0.0, -0.06006, -0.0, -0.0, -0.06006, 0.0, -0.0, -0.00337, -0.0, -0.0, 0.04794, -0.0, 0.0, 0.0176, -0.0, -0.0, 0.06505, -0.0, -0.0, 0.06505, -0.0, -0.0, 0.0176, -0.0, 0.0, 0.04794, -0.0, -0.0, -0.00337, -0.0, -0.0, -0.06006, -0.0, -0.0, -0.05598, -0.0, -0.0, -0.05598, -0.0, -0.0, -0.05598, -0.0, -0.0, -0.05598]]), "B2u":array([[0.01279, 0.01, 0.0, -0.0163, 0.08223, 0.0, 0.00519, 0.0871, 0.0, 0.02245, 0.01336, 0.0, 0.01304, -0.00966, -0.0, 0.01304, 0.00966, -0.0, 0.02245, -0.01336, -0.0, 0.00519, -0.0871, 0.0, -0.0163, -0.08223, 0.0, 0.01279, -0.01, 0.0, 0.01279, 0.01, -0.0, -0.0163, 0.08223, -0.0, 0.00519, 0.0871, -0.0, 0.02245, 0.01336, 0.0, 0.01304, -0.00966, 0.0, 0.01304, 0.00966, 0.0, 0.02245, -0.01336, -0.0, 0.00519, -0.0871, -0.0, -0.0163, -0.08223, -0.0, 0.01279, -0.01, -0.0, 0.01461, -0.01412, -0.0, 0.01461, 0.01412, -0.0, 0.01461, -0.01412, 0.0, 0.01461, 0.01412, 0.0], [-0.03431, 0.04564, 0.0, 0.03465, 0.0356, 0.0, 0.04674, -0.01502, 0.0, 0.00689, -0.02907, -0.0, -0.02434, -0.00306, -0.0, -0.02434, 0.00306, -0.0, 0.00689, 0.02907, 0.0, 0.04674, 0.01502, 0.0, 0.03465, -0.0356, 0.0, -0.03431, -0.04564, -0.0, -0.03431, 0.04564, 0.0, 0.03465, 0.0356, -0.0, 0.04674, -0.01502, -0.0, 0.00689, -0.02907, -0.0, -0.02434, -0.00306, 0.0, -0.02434, 0.00306, 0.0, 0.00689, 0.02907, 0.0, 0.04674, 0.01502, -0.0, 0.03465, -0.0356, -0.0, -0.03431, -0.04564, -0.0, -0.03334, 0.02158, -0.0, -0.03334, -0.02158, -0.0, -0.03334, 0.02158, 0.0, -0.03334, -0.02158, 0.0]]), "B3u":array([[0.00605, 0.04809, 0.0, 0.04863, 0.06963, 0.0, 0.04939, 0.04311, 0.0, 0.00026, 0.01253, 0.0, -0.01164, -0.01524, 0.0, 0.01164, -0.01524, -0.0, -0.00026, 0.01253, -0.0, -0.04939, 0.04311, -0.0, -0.04863, 0.06963, -0.0, -0.00605, 0.04809, -0.0, 0.00605, 0.04809, -0.0, 0.04863, 0.06963, -0.0, 0.04939, 0.04311, -0.0, 0.00026, 0.01253, -0.0, -0.01164, -0.01524, -0.0, 0.01164, -0.01524, 0.0, -0.00026, 0.01253, 0.0, -0.04939, 0.04311, 0.0, -0.04863, 0.06963, 0.0, -0.00605, 0.04809, 0.0, -0.01771, 0.00778, -0.0, 0.01771, 0.00778, 0.0, -0.01771, 0.00778, 0.0, 0.01771, 0.00778, -0.0], [-0.00202, -0.05311, 0.0, -0.01332, 0.0266, 0.0, -0.00753, 0.06593, 0.0, 0.01512, 0.00187, 0.0, 0.02552, 0.02034, 0.0, -0.02552, 0.02034, -0.0, -0.01512, 0.00187, -0.0, 0.00753, 0.06593, -0.0, 0.01332, 0.0266, -0.0, 0.00202, -0.05311, -0.0, -0.00202, -0.05311, -0.0, -0.01332, 0.0266, -0.0, -0.00753, 0.06593, -0.0, 0.01512, 0.00187, -0.0, 0.02552, 0.02034, 0.0, -0.02552, 0.02034, -0.0, -0.01512, 0.00187, 0.0, 0.00753, 0.06593, 0.0, 0.01332, 0.0266, 0.0, 0.00202, -0.05311, 0.0, 0.00683, -0.05243, 0.0, -0.00683, -0.05243, -0.0, 0.00683, -0.05243, -0.0, -0.00683, -0.05243, 0.0]])}, database_path = 'porphycene_scsd_20231020.pkl', maxdist = 1.75, smarts = '[C,c]1~[C,c]~[C,c](~[N,n]2)~[C,c]~[C,c]~[C,c](~2)~[C,c](~[N,n]6)~[C,c]~[C,c]~[C,c](~6)~[C,c]~[C,c]~[C,c](~[N,n]7)~[C,c]~[C,c]~[C,c](~7)~[C,c](~[N,n]8)~[C,c]~[C,c]~[C,c](~1)~8')
model_objs_dict['DADCAM'] = scsd_model('DADCAM', array([['-1.7145', '-0.0', '2.2778', 'C'], ['-0.7186', '-0.0', '1.2796', 'C'], ['4.1507', '0.0', '-2.1643', 'C'], ['-3.4065', '-0.0', '0.5302', 'C'], ['2.8242', '0.0', '-1.8426', 'C'], ['2.4246', '0.0', '-0.4926', 'C'], ['-4.1507', '0.0', '-2.1643', 'C'], ['3.0162', '-0.0', '1.9036', 'C'], ['-3.0162', '-0.0', '1.9036', 'C'], ['5.1211', '0.0', '-1.1611', 'C'], ['-4.7604', '-0.0', '0.1484', 'C'], ['-2.8242', '0.0', '-1.8426', 'C'], ['-2.4246', '0.0', '-0.4926', 'C'], ['-1.0969', '0.0', '-0.0397', 'C'], ['-5.1211', '0.0', '-1.1611', 'C'], ['1.0969', '0.0', '-0.0397', 'C'], ['3.4065', '-0.0', '0.5302', 'C'], ['4.7604', '-0.0', '0.1484', 'C'], ['1.7145', '-0.0', '2.2778', 'C'], ['0.7186', '-0.0', '1.2796', 'C'], ['0.0', '0.0', '-0.8784', 'O']]), 'C2v' , pca = {"A1":array([[0.0388, 0.0, 0.01704, 0.00334, -0.0, 0.03666, -0.03038, 0.0, -0.02387, 0.02931, 0.0, -0.00606, -0.01487, -0.0, 0.01562, -0.04329, -0.0, 0.00755, 0.03038, 0.0, -0.02387, -0.06048, 0.0, -0.0072, 0.06048, 0.0, -0.0072, -0.07104, 0.0, -0.0549, 0.08391, 0.0, -0.03129, 0.01487, -0.0, 0.01562, 0.04329, -0.0, 0.00755, -0.00115, -0.0, 0.02955, 0.07104, 0.0, -0.0549, 0.00115, -0.0, 0.02955, -0.02931, 0.0, -0.00606, -0.08391, 0.0, -0.03129, -0.0388, -0.0, 0.01704, -0.00334, -0.0, 0.03666, -0.0, -0.0, 0.03382], [-0.05684, 0.0, -0.01564, 0.01094, 0.0, -0.02529, -0.07009, 0.0, 0.00141, 0.06972, 0.0, -0.00732, -0.0788, 0.0, -0.01523, 0.03451, -0.0, -0.00243, 0.07009, 0.0, 0.00141, 0.03238, -0.0, 0.00708, -0.03238, -0.0, 0.00708, -0.04454, -0.0, 0.0321, 0.02641, -0.0, 0.04244, 0.0788, 0.0, -0.01523, -0.03451, -0.0, -0.00243, 0.01072, 0.0, -0.01576, 0.04454, -0.0, 0.0321, -0.01072, 0.0, -0.01576, -0.06972, 0.0, -0.00732, -0.02641, -0.0, 0.04244, 0.05684, 0.0, -0.01564, -0.01094, 0.0, -0.02529, 0.0, 0.0, -0.00269]]), "A2":array([[-0.0, 0.02871, -0.0, -0.0, 0.01385, 0.0, 0.0, 0.10615, -0.0, -0.0, 0.04966, 0.0, -0.0, -0.041, 0.0, 0.0, -0.03573, -0.0, 0.0, -0.10615, 0.0, 0.0, -0.04595, 0.0, 0.0, 0.04595, -0.0, 0.0, 0.09202, -0.0, 0.0, 0.06746, 0.0, -0.0, 0.041, -0.0, 0.0, 0.03573, 0.0, 0.0, 0.01946, -0.0, 0.0, -0.09202, 0.0, 0.0, -0.01946, 0.0, 0.0, -0.04966, -0.0, 0.0, -0.06746, -0.0, -0.0, -0.02871, -0.0, 0.0, -0.01385, -0.0, 0.0, -0.0, 0.0], [-0.0, 0.10541, -0.0, -0.0, 0.09436, -0.0, -0.0, -0.01991, -0.0, 0.0, -0.01325, 0.0, -0.0, 0.02485, 0.0, -0.0, -0.01819, 0.0, -0.0, 0.01991, 0.0, -0.0, -0.03294, 0.0, -0.0, 0.03294, -0.0, -0.0, 0.01219, -0.0, -0.0, -0.08409, 0.0, -0.0, -0.02485, -0.0, -0.0, 0.01819, -0.0, 0.0, 0.09481, 0.0, -0.0, -0.01219, 0.0, 0.0, -0.09481, -0.0, -0.0, 0.01325, -0.0, -0.0, 0.08409, -0.0, -0.0, -0.10541, -0.0, 0.0, -0.09436, 0.0, 0.0, -0.0, -0.0]]), "B1":array([[0.05901, -0.0, -0.03296, -0.00075, -0.0, -0.05378, -0.04792, 0.0, 0.01091, -0.00374, 0.0, -0.01221, 0.00015, 0.0, 0.01809, 0.021, 0.0, 0.01532, -0.04792, 0.0, -0.01091, 0.02034, 0.0, 0.02605, 0.02034, -0.0, -0.02605, -0.07801, -0.0, 0.0038, -0.05465, -0.0, -0.00149, 0.00015, -0.0, -0.01809, 0.021, -0.0, -0.01532, 0.05842, -0.0, -0.0556, -0.07801, -0.0, -0.0038, 0.05842, 0.0, 0.0556, -0.00374, 0.0, 0.01221, -0.05465, 0.0, 0.00149, 0.05901, 0.0, 0.03296, -0.00075, 0.0, 0.05378, 0.05231, -0.0, -0.0], [0.04025, 0.0, -0.01814, 0.11516, -0.0, 0.0524, -0.03063, 0.0, 0.00829, -0.00291, -0.0, -0.00602, -0.00197, 0.0, 0.01069, 0.0126, -0.0, 0.00846, -0.03063, -0.0, -0.00829, 0.01279, -0.0, 0.01917, 0.01279, 0.0, -0.01917, -0.04938, 0.0, 0.00359, -0.03444, -0.0, -0.0029, -0.00197, -0.0, -0.01069, 0.0126, 0.0, -0.00846, -0.00404, -0.0, 0.10022, -0.04938, -0.0, -0.00359, -0.00404, 0.0, -0.10022, -0.00291, 0.0, 0.00602, -0.03444, -0.0, 0.0029, 0.04025, -0.0, 0.01814, 0.11516, 0.0, -0.0524, -0.11485, 0.0, -0.0]]), "B2":array([[-0.0, 0.0082, 0.0, -0.0, -0.03695, 0.0, 0.0, 0.05017, -0.0, 0.0, 0.01317, 0.0, 0.0, -0.06192, -0.0, 0.0, -0.0387, 0.0, 0.0, 0.05017, 0.0, 0.0, 0.03504, -0.0, 0.0, 0.03504, -0.0, 0.0, 0.10256, 0.0, 0.0, 0.04087, -0.0, 0.0, -0.06192, -0.0, 0.0, -0.0387, 0.0, 0.0, -0.066, 0.0, 0.0, 0.10256, 0.0, -0.0, -0.066, 0.0, 0.0, 0.01317, 0.0, 0.0, 0.04087, 0.0, 0.0, 0.0082, 0.0, 0.0, -0.03695, -0.0, -0.0, -0.09287, 0.0], [-0.0, 0.04258, 0.0, 0.0, 0.05638, 0.0, 0.0, 0.06118, 0.0, 0.0, -0.04425, -0.0, 0.0, -0.05314, -0.0, 0.0, -0.01964, 0.0, 0.0, 0.06118, 0.0, 0.0, -0.02133, -0.0, 0.0, -0.02133, -0.0, 0.0, 0.04067, 0.0, 0.0, -0.11163, -0.0, 0.0, -0.05314, -0.0, 0.0, -0.01964, 0.0, -0.0, 0.03368, -0.0, 0.0, 0.04067, 0.0, 0.0, 0.03368, 0.0, 0.0, -0.04425, -0.0, 0.0, -0.11163, -0.0, 0.0, 0.04258, 0.0, 0.0, 0.05638, 0.0, 0.0, 0.03103, 0.0]])}, database_path = 'DADCAM_scsd_20231020.pkl', maxdist = 1.75, mondrian_limits = [-1, -1], smarts = '[O]1~[C,c]2~[C,c]3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3~[C,c]~[C,c]~[C,c]~2~[C,c]2~[C,c]~[C,c]~[C,c]3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3~[C,c]~1~2')
model_objs_dict['bodipy'] = scsd_model('bodipy', array([['0.0', '0.0', '-1.2741', 'B'], ['2.554', '-0.0', '1.497', 'C'], ['3.3646', '-0.0', '0.3858', 'C'], ['2.5162', '0.0', '-0.7301', 'C'], ['-2.5162', '0.0', '-0.7301', 'C'], ['-3.3646', '-0.0', '0.3858', 'C'], ['-2.554', '-0.0', '1.497', 'C'], ['-0.0', '-0.0', '1.7428', 'C'], ['1.2126', '-0.0', '1.04', 'C'], ['-1.2126', '-0.0', '1.04', 'C'], ['0.0', '1.1244', '-2.0829', 'F'], ['0.0', '-1.1244', '-2.0829', 'F'], ['1.2315', '0.0', '-0.3442', 'N'], ['-1.2315', '0.0', '-0.3442', 'N']]), 'C2v' , pca = {"A1":array([[-0.0, -0.0, -0.00079, -0.08034, 0.0, -0.03055, -0.10697, -0.0, 0.01283, -0.11562, -0.0, 0.09362, 0.11562, -0.0, 0.09362, 0.10697, -0.0, 0.01283, 0.08034, 0.0, -0.03055, 0.0, 0.0, -0.04871, -0.02637, 0.0, -0.04429, 0.02637, 0.0, -0.04429, 0.0, -0.03313, -0.00092, 0.0, 0.03313, -0.00092, -0.04768, 0.0, -0.00633, 0.04768, 0.0, -0.00633], [-0.0, -0.0, 0.06603, 0.05244, 0.0, -0.04488, 0.01881, 0.0, -0.00596, 0.03014, -0.0, 0.03977, -0.03014, -0.0, 0.03977, -0.01881, 0.0, -0.00596, -0.05244, 0.0, -0.04488, 0.0, 0.0, -0.20405, 0.00995, 0.0, -0.06243, -0.00995, 0.0, -0.06243, -0.0, 0.07375, 0.12946, -0.0, -0.07375, 0.12946, 0.00219, -0.0, 0.01393, -0.00219, -0.0, 0.01393]]), "A2":array([[0.0, -0.0, -0.0, -0.0, 0.18882, 0.0, 0.0, -0.05652, 0.0, 0.0, -0.11054, -0.0, 0.0, 0.11054, 0.0, 0.0, 0.05652, -0.0, -0.0, -0.18882, -0.0, -0.0, -0.0, 0.0, -0.0, 0.07423, -0.0, -0.0, -0.07423, 0.0, 0.00754, 0.0, -0.0, -0.00754, 0.0, 0.0, 0.0, -0.06235, 0.0, 0.0, 0.06235, -0.0], [0.0, -0.0, 0.0, -0.0, -0.02157, 0.0, 0.0, 0.15846, 0.0, 0.0, -0.08617, 0.0, -0.0, 0.08617, -0.0, 0.0, -0.15846, -0.0, 0.0, 0.02157, -0.0, -0.0, -0.0, 0.0, 0.0, -0.06073, 0.0, 0.0, 0.06073, -0.0, 0.05074, 0.0, -0.0, -0.05074, 0.0, 0.0, -0.0, -0.12234, 0.0, -0.0, 0.12234, -0.0]]), "B1":array([[-0.00342, -0.0, 0.0, 0.07655, -0.0, -0.03377, -0.04615, 0.0, 0.0499, -0.1175, -0.0, 0.15125, -0.1175, 0.0, -0.15125, -0.04615, 0.0, -0.0499, 0.07655, -0.0, 0.03377, 0.09883, -0.0, 0.0, 0.05761, -0.0, -0.01759, 0.05761, -0.0, 0.01759, -0.02344, 0.0, 0.0, -0.02344, 0.0, -0.0, 0.00566, 0.0, -0.0213, 0.00566, -0.0, 0.0213], [-0.05674, -0.0, -0.0, 0.02847, -0.0, 0.10213, -0.02191, 0.0, 0.03898, -0.00455, 0.0, -0.07144, -0.00455, -0.0, 0.07144, -0.02191, 0.0, -0.03898, 0.02847, 0.0, -0.10213, 0.10507, -0.0, -0.0, 0.05772, -0.0, 0.04888, 0.05772, -0.0, -0.04888, -0.07686, 0.0, 0.0, -0.07686, 0.0, -0.0, -0.00758, -0.0, 0.04362, -0.00758, 0.0, -0.04362]]), "B2":array([[-0.0, -0.0081, -0.0, 0.0, 0.03635, -0.0, -0.0, -0.07701, 0.0, 0.0, -0.11961, -0.0, -0.0, -0.11961, 0.0, -0.0, -0.07701, 0.0, 0.0, 0.03635, 0.0, 0.0, 0.10494, 0.0, 0.0, 0.05212, 0.0, 0.0, 0.05212, -0.0, -0.0, 0.10139, 0.0574, 0.0, 0.10139, -0.0574, -0.0, -0.04187, -0.0, 0.0, -0.04187, -0.0], [0.0, -0.25737, -0.0, -0.0, 0.0274, 0.0, -0.0, 0.1227, -0.0, 0.0, 0.00743, 0.0, -0.0, 0.00743, -0.0, -0.0, 0.1227, 0.0, -0.0, 0.0274, -0.0, -0.0, -0.01372, -0.0, -0.0, -0.05334, 0.0, 0.0, -0.05334, 0.0, 0.0, 0.09264, -0.00248, 0.0, 0.09264, 0.00248, -0.0, -0.06091, 0.0, -0.0, -0.06091, -0.0]])}, database_path = 'bodipy_scsd_20240128.pkl', maxdist = 1.75, smarts = '[N,n]12~[C,c]~[C,c]~[C,c]~[C,c](~1)~[C,c]~[C,c]1~[C,c]~[C,c]~[C,c]~[N,n](~1)~B(~2)(~F)~F')
model_objs_dict['phenanthroline'] = scsd_model('phenanthroline', array([['2.817', '-0.0', '0.8543', 'C'], ['3.5211', '0.0', '-0.2959', 'C'], ['2.8582', '0.0', '-1.5065', 'C'], ['1.4842', '0.0', '-1.563', 'C'], ['-1.4842', '0.0', '-1.563', 'C'], ['-2.8582', '0.0', '-1.5065', 'C'], ['-3.5211', '0.0', '-0.2959', 'C'], ['-2.817', '-0.0', '0.8543', 'C'], ['-0.6689', '-0.0', '2.0623', 'C'], ['0.6689', '-0.0', '2.0623', 'C'], ['1.4032', '-0.0', '0.8446', 'C'], ['0.7268', '0.0', '-0.3957', 'C'], ['-0.7268', '0.0', '-0.3957', 'C'], ['-1.4032', '-0.0', '0.8446', 'C']]), 'C2v' , maxdist = 1.75, mondrian_limits = [-1.0, -1.0], smarts = '[C,c]1~[C,c]~[C,c]~[C,c]2~[C,c](~[C,c]~1)~[C,c]~[C,c]~[C,c]1~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~1')
