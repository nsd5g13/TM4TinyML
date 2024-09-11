# Convert the post-training clause expressions to a REDRESS encoded format, stored in text

import sys, os

no_clauses = int(sys.argv[1])
vanillaORdiet = sys.argv[2]

if vanillaORdiet.lower() == 'vanilla':
	f = open(r"../VanillaTM_training/log/actions.txt", "r")
elif vanillaORdiet.lower() == 'diet':
	f = open(r"../DIET_training/log/actions.txt", "r")
else:
	print("Choose the action file produced by either vanilla TM (vanilla) or DIET (diet)")
f_lines = f.readlines()

redress_codes = []
no_classes = int(len(f_lines)/no_clauses)
includes_classes = [0] * no_classes
last_clause_bit = '0'
for i in range(no_classes):
	for j in range(no_clauses):
		clause_bit = str(1 - int(last_clause_bit))
		org_code = f_lines[i*no_clauses+j].replace('\n','')
		if j < no_clauses/2:
			PosPolarity = True
		else:
			PosPolarity = False
		for k, include in enumerate(org_code):
			if include == '1':
				if PosPolarity == True:
					redress_code = '0'
				else:
					redress_code = '1'

				redress_code = redress_code + clause_bit

				if k < len(org_code)/2:
					complement = False
					feature_index = k
				else:
					complement = True
					feature_index = int(k - len(org_code)/2)

				redress_code = redress_code + format(feature_index, '013b')
				
				if complement == False:
					redress_code = redress_code + '0'
				else:
					redress_code = redress_code + '1'
				
				redress_codes.append(int(redress_code,2))

				includes_classes[i] = includes_classes[i] + 1
		last_clause_bit = clause_bit

if not os.path.exists(r"redress"):
	os.makedirs(r"redress")

encode_file = open(r"redress/encoded_include.txt", "w")
for redress_code in redress_codes:
	encode_file.write(str(redress_code)+'\n')
encode_file.close()

no_includes_file = open(r"redress/no_includes.txt", "w")
for includes_class in includes_classes:
	no_includes_file.write(str(includes_class)+'\n')
no_includes_file.close()