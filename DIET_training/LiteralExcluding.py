import re
import numpy as np

# Capture common literals included in both postive and negative clauses of a class as the insignificant literals
def InsignificantLiterals(expr_file_name):
	InsigLit = []
	expr_file = open(expr_file_name, "r")
	lines = expr_file.readlines()
	pattern = re.compile("Class [0-9]+ Positive Clauses:")
	class_start_line = []
	for i, each in enumerate(lines):
		if pattern.match(each):
			InsigLit.append([])
			class_start_line.append(i)
	class_start_line.append(len(lines))

	for i, each in enumerate(class_start_line[:-1]):
		nexteach = class_start_line[i+1]
		PostiveClause = False
		PosClauseLit = []
		NegClauseLit = []
		for line in lines[each:nexteach]:
			if "Positive Clauses" in line:
				PostiveClause = True
			elif "Negative Clauses" in line:
				PostiveClause = False
			
			if "Clause" not in line and line != '\n' and line !='':
				literals = line.replace(' ', '').replace('\n','').split('&')
				if PostiveClause == True:
					PosClauseLit.extend(literals)
				else:
					NegClauseLit.extend(literals)
		PosClauseLit = set(PosClauseLit)
		NegClauseLit = set(NegClauseLit)
		Overlaps = list(PosClauseLit & NegClauseLit)
		#Overlaps = PosClauseLit.(NegClauseLit)
		InsigLit[i].extend(Overlaps)

	InsigLit_file = open(r"log/excluded_literals.txt", "w")
	for i, each in enumerate(InsigLit):
		InsigLit_file.write("Class %d: \n" %i)
		for lit in each:
			InsigLit_file.write("%s " %lit)
		InsigLit_file.write("\n\n")
	InsigLit_file.close()

	return InsigLit

# Read the state file and prune the insignificant lieterals in the states
# States of all literals are stored as follows in states.txt: 
# 32b 32b 32b 32b 32b 32b 32b x31-x30-x29-...-x0
# 32b 32b 32b 32b 32b 32b 32b x63-x62-x61-...-x32
# ...
# And then for the next clause with larger index
# Prune by only setting the MSB of TA state as 0
def Prune(InsigLit, state_file_name, number_of_literals, number_of_clauses):
	state_file = open(state_file_name, "r")
	lines = state_file.readlines()
	all_states = []
	# Read out all states from the file
	for i in range(int(len(lines)/3)):
		all_states.append([])
		line = lines[3*i+1]
		states = line.replace('[', '').replace(']', '').replace('\n', '').split()
		all_states[i].extend(states)
		
	# Prune the insignificant literals
	new_all_states = all_states
	for i, class_states in enumerate(all_states):
		clause_states = []
		for j, states in enumerate(class_states):
			if j%8 == 7: # TA state is given as 8, indicating 8 previous states are stored for each literals
				clause_states.append(states)
		clause_states = np.array(clause_states).reshape(number_of_clauses, int(len(clause_states)/number_of_clauses))
	
		new_clause_states = clause_states
		for j, each_clause_states in enumerate(clause_states):
			for k, each_grp_states in enumerate(each_clause_states):
				if each_grp_states != '0':
					states_bin = format(int(each_grp_states), '032b')
					new_states_bin = format(0, '032b')
					for l, state in enumerate(states_bin):
						if state == '1':
							literal_no = (k+1)*32 - l - 1
							if literal_no < number_of_literals/2:
								literal = 'x' + str(literal_no)
							else:
								literal = '~x' + str(int(literal_no-number_of_literals/2))	
							
							if literal not in InsigLit[i]:
								new_states_bin = new_states_bin[:l] + '1' + new_states_bin[l+1:]
					new_clause_states[j][k] = str(int(new_states_bin,2))

		new_clause_states = [j for sub in new_clause_states for j in sub]
	
		for j in range(len(new_all_states[i])):
			if j%8 == 7:
				new_all_states[i][j] = new_clause_states[int(np.floor(j/8))]
	
	for i in range(len(new_all_states)):
		for j in range(len(new_all_states[i])):
			new_all_states[i][j] = int(new_all_states[i][j])

	return new_all_states