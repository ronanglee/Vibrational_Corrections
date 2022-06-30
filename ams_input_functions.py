import os
import multiprocessing as mp
import numpy as np
import numpy.polynomial.polynomial as poly
import itertools as it
from numpy import linalg as la
from atomic_masses import atomic_masses

def worker(index, job):  # Function to take a list of run-files and run successively
    for o in job:
        os.system("bash " + o)

def read_input_ams():
    user_input_opt = ""  # Name of input file from user containing the optimization
    user_input_freq = ""  # Name of input file from user containing the frequency calculation of
    user_input_nmr = "" # Name of input file from user containing the NMR options i.e shieldings
    number_of_atoms = ""
    charge = 0  # Charge of the molecule
    atomtypes = ""
    optimize = ""
    linear = ""
    points = 0
    multiplicity = 0  # Spin multiplicity of the molecule
    properties = ""  # Which molecular property to vibrationally average
    temperature = 0  # Absolute temperature in Kelvin
    file_name_start = ""  # Name of the first input file containing the coordinates, Dalton
    file_name_calc_opt_1 = ""  # Name of the file containing the calculation options for the geometry optimizations, Dalton
    file_name_calc_opt_2 = ""  # Name of the file containing the calculation options for the frequency calculation, Dalton
    file_name_calc_opt_3 = ""  # Name of the file containing the calculation options, Dalton
    # for the calculation of the property that's supposed to be calculated, Dalton
    text_line3 = "" # user input, Dalton
    text_line4 = "" # user input, Dalton
    h = 0  # Step length for derivative calculations, Gaussian
    functional_p = ""  # Functional for property calculations, Gaussian
    functional_f = ""  # Functional for frequency calculations, Gaussian
    basis_set_p = ""  # Basis set for property calculations, Gaussian
    basis_set_f = ""  # Basis set for frequency calculations, Gaussian
    diff_method = ""  # Method to calculate derivatives (linear/polynomial)
    mode = ""  # Mode for cubic force constants by taking derivatives of harmonic frequencies or force constants
    order = 0  # Polynomial fit order
    anharm = ""  # Whether or not to use Gaussian's built-in analytical function for calculating cubic force constants
    anharmq = ""  # Whether or not to use Gaussian's built-in analytical function for calculating mean displacements
    method_geo = ""  # Functional for geometry calculations, CFOUR
    basis_geo = ""  # Basis set for geometry calculations, CFOUR
    method_prop = ""  # Functional for property calculations, CFOUR
    basis_prop = ""  # Basis set for property calculations, CFOUR
    stepsize = 0  # Step length for derivative calculations, CFOUR
    stepnumber = 0  # Number of steps for derivative calculations, CFOUR
    run = ""
    old_h = "y"

    with open("general_input.txt", "r") as my_input:  # Read job information file
        job_input = my_input.readlines()
        for line_i in job_input:
            if "Software: " in line_i:
                software = line_i.split()[-1]
            elif "User added Optimization info: " in line_i:
                user_input_opt = line_i.split()[-1]
            elif "User added Frequency calculation info: " in line_i:
                user_input_freq = line_i.split()[-1]
            elif "User added NMR properties info: " in line_i:
                user_input_nmr = line_i.split()[-1]
            elif "Input File name: " in line_i:
                file_name_start = line_i.split()[-1]
            elif "Linear molecule (Y/N): " in line_i:
                linear = line_i.split()[-1]
            elif "File name - calculation options 1: " in line_i:
                file_name_calc_opt_1 = line_i.split()[-1]
            elif "File name - calculation options 2: " in line_i:
                file_name_calc_opt_2 = line_i.split()[-1]
            elif "File name - calculation options 3: " in line_i:
                file_name_calc_opt_3 = line_i.split()[-1]
            elif "Text in line 3: " in line_i:
                text_line3 = line_i.strip("Text in line 3: ")
            elif "Text in line 4: " in line_i:
                text_line4 = line_i.strip("Text in line 4: ")
            elif "Atomtypes:" in line_i:
                atomtypes = "Atomtypes=" + line_i.split()[-1]
            elif "Number of atoms: " in line_i:
                number_of_atoms = int(line_i.split()[-1])
            elif "Optimize geometry (Y/N): " in line_i:
                optimize = line_i.split()[-1]
            elif "Charge:" in line_i:
                charge = "    Charge " + line_i.split()[-1]
            elif "Generators: " in line_i:
                generators = "Generators=" + line_i.split()[-1]
            elif "Number of points (3/5): " in line_i:
                points += int(line_i.split()[-1])
            elif "Multiplicity: " in line_i:
                multiplicity = int(line_i.split()[-1])
            elif "Molecular property (NMR, J_FC, J_SD, J_SO): " in line_i:
                properties = line_i.split()[-1]
            elif "Temperature (K): " in line_i:
                temperature = float(line_i.split()[-1])
            elif "h: " in line_i:
                h = float(line_i.split()[-1])
            elif "Functional property: " in line_i:
                functional_p = line_i.split()[-1]
            elif "Functional freq/opt: " in line_i:
                functional_f = line_i.split()[-1]
            elif "Basis set property: " in line_i:
                basis_set_p = line_i.split()[-1]
            elif "Basis set freq/opt: " in line_i:
                basis_set_f = line_i.split()[-1]
            elif "Optimize geometry (Y/N): " in line_i:
                optimize = line_i.split()[-1]
            elif "Differentiation method (linear/polynomial): " in line_i:
                diff_method = line_i.split()[-1].lower()
            elif "Derivative mode (freq/fc): " in line_i:
                mode = line_i.split()[-1].lower()
            elif "Polynomial fit order: " in line_i:
                order = int(line_i.split()[-1])
            elif "Anharmonic (Y/N): " in line_i:
                anharm = line_i.split()[-1].lower()
            elif "Anharmonic Q (Y/N): " in line_i:
                anharmq = line_i.split()[-1].lower()
            elif "method_geo" in line_i:
                method_geo = line_i.split()[-1]
            elif "basis_geo" in line_i:
                basis_geo = line_i.split()[-1]
            elif "method_prop" in line_i:
                method_prop = line_i.split()[-1]
            elif "basis_prop" in line_i:
                basis_prop = line_i.split()[-1]
            elif "stepnumber" in line_i:
                stepnumber += int(line_i[12:14])
            elif "Run new calculations? (Y/N):" in line_i:
                run = line_i.split()[-1].lower()
                if run == "n" or run == "no":
                    optimize = "n"
            elif "Old step method? (Y/N):" in line_i:
                old_h = line_i.split()[-1].lower()

    return (software, user_input_opt, user_input_freq, user_input_nmr, file_name_start, linear, optimize, file_name_calc_opt_1, file_name_calc_opt_2, file_name_calc_opt_3, text_line3, text_line4,
            atomtypes, number_of_atoms, charge, generators, multiplicity, properties, temperature,
            h, functional_p, functional_f, basis_set_p, basis_set_f, diff_method, points, mode, order, anharm,
            anharmq, method_geo, basis_geo, method_prop, basis_prop, stepsize, stepnumber, run, old_h)

def read_inputgeometry_ams(file_name_start, number_of_atoms):
    (x_e, y_e, z_e, lines_unformatted) = ([], [], [], [])  # Empty lists to append input coordinates
    atoms = []  # Empty list to append atom symbols
    atomstypes_int = ""
    with open(file_name_start, "r") as my_input:
        job_input = my_input.readlines()
        for line_i in job_input:
            if "Atomtypes=" in line_i:
                atomstypes_int = int(line_i.split()[0].strip("Atomtypes="))
    with open(file_name_start, "r") as my_input:
        start = 2
        my_text = my_input.readlines()
        for i, line in enumerate(my_text):
            if "Atomtypes=" in line:
                start += i
        for i, line in enumerate(my_text[start:]):
            lines_unformatted.append(line)

        del lines_unformatted[1]
        lines_unformatted.remove("FINISH\n")

        k = []
        l = []
        lines = []
        for i in lines_unformatted:
            j = i.replace(' ', '', 1)
            k.append(j)
        for i in k:
            m = i.replace('   ', '    ', 2)
            l.append(m)
        for i in l:
            q = i.replace('             ', '', 3)   # reformat input coordinates for AMS input file
            lines.append(q)

        for line in lines_unformatted[0:number_of_atoms]:
            atoms.append(line.split()[0])
            x_e.append(float(line.split()[1]))
            y_e.append(float(line.split()[2]))
            z_e.append(float(line.split()[3]))

    return atomstypes_int, lines, atoms, x_e, y_e, z_e

def write_input_ams(file_name_input, charge, lines, number_of_atoms, file_list, user_input_opt):
    with open (file_name_input,"w") as input_file1:
        input_file1.write("#!/bin/sh"), input_file1.write("\n")
        input_file1.write("\n")
        input_file1.write("$AMSBIN/ams <<eor"), input_file1.write("\n")
        input_file1.write("System"), input_file1.write("\n")
        input_file1.write("  atoms"), input_file1.write("\n")
        for i in lines[:number_of_atoms]:  # Write lines of atoms and coordinates
            input_file1.write("     " + i)
        input_file1.write("  end"), input_file1.write("\n")
        input_file1.write(charge), input_file1.write("\n")
        input_file1.write("end")
        input_file1.write("\n")
        with open(user_input_opt, "r") as my_input_w2:
            job_input_w2 = my_input_w2.readlines()
            for line_i in job_input_w2:
                    input_file1.write(line_i)
        file_list.append(file_name_input) # Add newly written file to list of filenames

    return file_list

def write_lines(file_name_start, number_of_atoms, atomstypes_int, atoms, x_e, y_e, z_e):
    with open(file_name_start, "r") as my_input:
        lines_test = []
        lines = []
        start_1 = 0
        start_2 = 0
        count = 0
        my_text = my_input.readlines()
        for i, line in enumerate(my_text):
            if "Atoms=" in line:
                start_1 += i
                start_2 += i
                break
        for _ in range(number_of_atoms+atomstypes_int):
            for line_a in my_text[start_1:]:
                if "Charge" in line_a:
                    lines.append(line_a)
                    start_1 += 1
                    break
                if not "Charge" in line_a:
                    lines.append(line_a)
                    start_1 += 1

        for _ in range(number_of_atoms+atomstypes_int):
            for line_a in my_text[start_2:]:
                if "Charge" in line_a:
                    start_2 += 1
                    break
                if not "Charge" in line_a:
                    lines_test.append(lines[_].replace(line_a, "{0:}  {1:< 20.15f}{2:< 20.15f}{3:< 20.15f}\n".format
                    (atoms[count], x_e[count], y_e[count], z_e[count])))
                    start_2 += 1
                    count += 1
                    break

    return lines, lines_test

def call_ams(workers, file_list, run):
    if run == "y":
        listt = []
        for j in range(len(file_list)):
            file_name1 = "run_ams_" + file_list[j].strip(".run") + ".tmp"
            os.system('sed "s/dummyCOM/'+file_list[j].strip(".run")+'/g" run_ams.tmp > '+file_name1)
            listt.append(file_name1)  # Generates run-file by replacing dummy variables with the right names and values
        job_array = [listt[s::workers] for s in range(workers)]  # Splits up list after how many CPU's are available
        processes = [mp.Process(target=worker, args=(s, job_array[s])) for s in range(workers)]  # Run program
        for p in processes:
            p.start()
        for p in processes:
            p.join()

def read_optimized_ams(output_file_name, number_of_atoms):
    (x_e, y_e, z_e, lines) = ([], [], [], [])  # Empty lists to append equilibrium coordinates
    with open(output_file_name, "r") as my_input_o:  # Read output files
        my_text_o = my_input_o.readlines()
        for j, line_o in enumerate(my_text_o):
            if "Optimized geometry:" in line_o:
                print("Geometry optimization succesful")
                start_o = j
        for j, line_o in enumerate(my_text_o):
            if j in range(start_o + 7, start_o + 7 + number_of_atoms):
                lines.append(line_o)

    for line in lines[0:number_of_atoms]:
        x_e.append(float(line.split()[2]))
        y_e.append(float(line.split()[3]))
        z_e.append(float(line.split()[4]))

    return x_e, y_e, z_e  # Given in angstroms

# def read_hessian(filename, atoms):
#     with open(filename, "r") as file_input:
#         file_lines = file_input.readlines()
#         divs = (len(atoms) * 3 - 1) // 6 + 1
#         hessian = np.zeros((len(atoms) * 3, len(atoms) * 3))
#         for idx, l in enumerate(file_lines):
#             if "Molecular Hessian (au)" in l:
#                 for idl, ll in enumerate(file_lines[idx:]):
#                     if "{0}    x      {0}    y      {0}    z      {1}   x      {1}   y      {1}   z".format(
#                             atoms[0], atoms[1]) in ll:
#                         for d in range(divs):
#                             for p in range(len(atoms) - 2 * d):
#                                 for a in range(3):
#                                     b = 0
#                                     while (a + p * 3) >= b and b < 6:
#                                         hessian[a + p * 3 + d * 6, b + d * 6] = \
#                                             file_lines[idx + idl + 2 + a + p * 4][8:].split()[b]
#                                         b += 1
#                             idl += (3 + 2 + 2 + (len(atoms) - 2 * d - 1) * 4)
#         for x in range(len(atoms) * 3):
#             for y in range(len(atoms) * 3):
#                 if x < y:
#                     hessian[x, y] = hessian[y, x]
#     return hessian

def check_freq_real_eq(output_file_name, number_of_atoms):
    (freq_check, lines_o) = ([], [])
    with open(output_file_name, "r") as my_input_o_freq:  # Read output files
        my_text_o = my_input_o_freq.readlines()
        for j, line_o in enumerate(my_text_o):
            if "Index   Frequency (cm-1)   Intensity (km/mol)" in line_o:
                start_o = j
        for j, line_o in enumerate(my_text_o):
            if j in range(start_o + 1, start_o + 1 + number_of_atoms):
                lines_o.append(line_o)
    for line in lines_o[0:number_of_atoms * 2]:
        freq_check.append(float(line.split()[1]))
    for freq in freq_check:
        if freq < 0:
            print("Imaginary frequency for equilibrium geometry")
            sys.exit(0)

def check_freq_real(output_file_name_list, number_of_atoms, number_of_freqs):
    (freq_check,lines_o) = ([],[])
    start_o = 0
    for j in range(len(output_file_name_list)):
        with open(output_file_name_list[j].strip("run") + "out", "r") as my_input_o_freq:  # Read frequency calculation output
            my_text_o = my_input_o_freq.readlines()
        for j, line_o in enumerate(my_text_o):
             if "Index   Frequency (cm-1)   Intensity (km/mol)" in line_o:
                start_o = j
        for j, line_o in enumerate(my_text_o):
            if j in range(start_o + 1, start_o + 1 + number_of_atoms):
                lines_o.append(line_o)
    for line in lines_o[0:number_of_freqs]:
           freq_check.append(float(line.split()[1]))
    for freq in freq_check:
        if freq < 0:
            print("Error: Imaginary frequencies in {0}".format(output_file_name))
            freq_check.remove(freq)

def read_and_count_freqs(output_file_name):
    frequencies = []
    with open(output_file_name, "r") as my_file:
        my_file_lines = my_file.readlines()
        start_lines = 0
        end_lines = 0
        for j, line in enumerate(my_file_lines):
            if " Index   Frequency (cm-1)   Intensity (km/mol)" in line:
                start_lines += j
            if "Zero-point energy (Hartree):" in line:
                end_lines += j
        for line in my_file_lines[start_lines+1:end_lines-1]:
            frequencies.append(float(line.split()[1]))
            number_of_freqs = len(frequencies)
	

    return number_of_freqs, frequencies

def read_normalcoordinates_ams(output_file_name, number_of_atoms):
    x_nm = {}
    y_nm = {}
    z_nm = {}
    for i in range(number_of_atoms):
        x_nm[i] = []  # Normal coordinates
        y_nm[i] = []
        z_nm[i] = []
    with open(output_file_name, "r") as my_input_n:  # Read frequency calculation output
        my_text_n = my_input_n.readlines()
        for m, line_n in enumerate(my_text_n):
            if "Index  Atom      ---- Displacements (x/y/z) ----" in line_n:  # Find and append normal coordinates
                start_n = m
                k = 0
                for line in my_text_n[start_n+1:start_n+1+number_of_atoms]:
                    x_nm[k].append(float(line.split()[2]))
                    y_nm[k].append(float(line.split()[3]))
                    z_nm[k].append(float(line.split()[4]))
                    k+=1

    return x_nm, y_nm, z_nm

def read_normalcoordinates_1_ams(output_file_name, number_of_atoms):
    x_nm_1 = []
    y_nm_1 = []
    z_nm_1 = []
    with open(output_file_name, "r") as my_input_n:  # Read frequency calculation output
        my_text_n = my_input_n.readlines()
        for m, line_n in enumerate(my_text_n):
            if "Index  Atom      ---- Displacements (x/y/z) ----" in line_n:  # Find and append normal coordinates
                start_n = m
                k = 0
                for line in my_text_n[start_n+1:start_n+1+number_of_atoms]:
                    x_nm_1.append(float(line.split()[2]))
                    y_nm_1.append(float(line.split()[3]))
                    z_nm_1.append(float(line.split()[4]))
                    k+=1

    return x_nm_1, y_nm_1, z_nm_1

def read_normalcoordinates_disp_ams(output_file_name_list, number_of_freqs, number_of_atoms):
    freqs = []
    x_s = []
    y_s = []
    z_s = []
    for j in range(len(output_file_name_list)):
        with open(output_file_name_list[j].strip("run") + "out", "r") as my_input_n:
        # with open(output_file_name_list, "r") as my_input_n:  # Read frequency calculation output
            my_text_n = my_input_n.readlines()
            for i in range(number_of_atoms):
                start_f = 0
            for m, line_n in enumerate(my_text_n):
                if "Index  Atom      ---- Displacements (x/y/z) ----" in line_n:  # Find and append normal coordinates
                    start_n = m
                    k = 0
                    for line in my_text_n[start_n + 1:start_n + 1 + number_of_atoms]:
                        x_s.append(float(line.split()[2]))
                        y_s.append(float(line.split()[3]))
                        z_s.append(float(line.split()[4]))
                        k += 1
            for j, line_o in enumerate(my_text_n):
                if "Index   Frequency (cm-1)   Intensity (km/mol)" in line_o:
                    start_f += j
                    for line in my_text_n[start_f + 1: start_f + 1 + number_of_freqs]:
                            freqs.append(float(line.split()[1]))
                    
    return x_s, y_s, z_s, freqs

def calc_reduced_masses(number_of_atoms, x_nm, y_nm, z_nm):
    frequency = {}
    reduced_masses_1 = []
    reduced_masses = {}
    for i in range(number_of_atoms):
        frequency[i] = x_nm[i] + y_nm[i] + z_nm[i]

        for j in frequency[i]:
            reduced_masses_1.append(j**2)
            if len(reduced_masses_1) == len(frequency[i]):
                reduced_masses[i] = 1/sum(reduced_masses_1)
                reduced_masses_1 = []
    return reduced_masses

def write_nc_ams(file_name_start, x_n, y_n, z_n, atoms):
    with open("../normalcoordinates.txt", "w") as my_nc:  # Write file with normal coordinates and atom symbols
        my_nc.write("Molecule: " + file_name_start + "\n")
        my_nc.write("Normal coordinates, x: \n")
        atom_string = ""
        for a in atoms:
            atom_string = atom_string + ("{0:>13}".format(a))
        my_nc.write(atom_string)
        my_nc.write("\n")
        for i in x_n:
            x_string = ""
            for a in i:
                x_string = x_string + "{0:13.5f}".format(a)
            my_nc.write(x_string)
            my_nc.write("\n")
        my_nc.write("\n")
        my_nc.write("Normal coordinates, y: \n")
        my_nc.write(atom_string)
        my_nc.write("\n")
        for i in y_n:
            y_string = ""
            for a in i:
                y_string = y_string + "{0:13.5f}".format(a)
            my_nc.write(y_string)
            my_nc.write("\n")
        my_nc.write("\n")
        my_nc.write("Normal coordinates, z: \n")
        my_nc.write(atom_string)
        my_nc.write("\n")
        for i in z_n:
            z_string = ""
            for a in i:
                z_string = z_string + "{0:13.5f}".format(a)
            my_nc.write(z_string)
            my_nc.write("\n")
        my_nc.write("\n")


def write_loop_ams(
        number_of_atoms, points, lines, x_e, y_e, z_e, x_n, y_n, z_n, step, atoms, charge,
        file_list, anharm, qi, frequencies, file_name_start, atomstypes_int, user_input_freq, user_input_nmr, h, old_h):
    file_list_freq = []  # Empty list to append file names of frequency calculation input files
    h_list = []
    with open("general_input.txt", "r") as my_input:  # Read job information file
        job_input = my_input.readlines()
        for line_i in job_input:
            if "User added Frequency calculation info: " in line_i:
                user_input_freq = line_i.split()[-1]
            elif "User added NMR properties info: " in line_i:
                user_input_nmr = line_i.split()[-1]
    for a in range(qi):# Loop normal modes four times for polynomial and twice for linear
        h = step * np.exp((-0.004 * step - 0.0003) * frequencies[a])
        if h < 0.01:
            h = 0.01
        if old_h == "y":
            h = step
        h_list.append(h)
        if points == 5:
            for _ in range(number_of_atoms):# Loop atoms defining new distorted coordinates
                lines.pop(0)
            ctr = 0
            (x, y, z) = ([], [], [])
            for b in range(number_of_atoms):
                x.append(x_e[b] - x_n[a, b] * 2 * h)
                y.append(y_e[b] - y_n[a, b] * 2 * h)
                z.append(z_e[b] - z_n[a, b] * 2 * h)
                ctr += 1
            (lines, lines_test) = write_lines(file_name_start, number_of_atoms, atomstypes_int, atoms, x, y, z)
            write_input_ams(
                            str(a) + "_minus2.run", charge, lines_test, number_of_atoms, file_list, user_input_nmr)

        for _ in range(number_of_atoms):
            lines.pop(0)
        ctr = 0
        (x, y, z) = ([], [], [])
        for b in range(number_of_atoms):
            x.append(x_e[b] - x_n[a, b] * h)
            y.append(y_e[b] - y_n[a, b] * h)
            z.append(z_e[b] - z_n[a, b] * h)
            ctr += 1
        (lines, lines_test) = write_lines(file_name_start, number_of_atoms, atomstypes_int, atoms, x, y, z)
        write_input_ams(
            str(a) + "_minus1.run", charge, lines_test, number_of_atoms, file_list, user_input_nmr)

        for _ in range(number_of_atoms):
            lines.pop(0)
        ctr = 0
        (x, y, z) = ([], [], [])
        for b in range(number_of_atoms):
            x.append(x_e[b] + x_n[a, b] * h)
            y.append(y_e[b] + y_n[a, b] * h)
            z.append(z_e[b] + z_n[a, b] * h)
            ctr += 1
        (lines, lines_test) = write_lines(file_name_start, number_of_atoms, atomstypes_int, atoms, x, y, z)
        write_input_ams(
            str(a) + "_plus1.run", charge, lines_test, number_of_atoms, file_list, user_input_nmr)

        if points == 5:
            for _ in range(number_of_atoms):
                lines.pop(0)
            ctr = 0
            (x, y, z) = ([], [], [])
            for b in range(number_of_atoms):
                x.append(x_e[b] + x_n[a, b] * 2 * h)
                y.append(y_e[b] + y_n[a, b] * 2 * h)
                z.append(z_e[b] + z_n[a, b] * 2 * h)
                ctr += 1
            (lines, lines_test) = write_lines(file_name_start, number_of_atoms, atomstypes_int, atoms, x, y, z)
            write_input_ams(
                str(a) + "_plus2.run", charge, lines_test, number_of_atoms, file_list, user_input_nmr)

    if anharm == "n" or anharm == "no" or anharm == "x":
        for a in range(qi):  # Loop normal modes for frequency calculations
            if points == 5:
                for _ in range(number_of_atoms):
                    lines.pop(0)
                ctr = 0
                (x, y, z) = ([], [], [])
                for b in range(number_of_atoms):
                    x.append(x_e[b] - x_n[a, b] * 2 * h)
                    y.append(y_e[b] - y_n[a, b] * 2 * h)
                    z.append(z_e[b] - z_n[a, b] * 2 * h)
                    ctr += 1
                (lines, lines_test) = write_lines(file_name_start, number_of_atoms, atomstypes_int, atoms, x, y, z)
                write_input_ams(
                    str(a) + "_freq_minus2.run", charge, lines_test, number_of_atoms, file_list_freq, user_input_freq)

            for _ in range(number_of_atoms):
                lines.pop(0)
            ctr = 0
            (x, y, z) = ([], [], [])
            for b in range(number_of_atoms):
                x.append(x_e[b] - x_n[a, b] * h)
                y.append(y_e[b] - y_n[a, b] * h)
                z.append(z_e[b] - z_n[a, b] * h)
                ctr += 1
            (lines, lines_test) = write_lines(file_name_start, number_of_atoms, atomstypes_int, atoms, x, y, z)
            write_input_ams(
                str(a) + "_freq_minus1.run", charge, lines_test, number_of_atoms, file_list_freq, user_input_freq)

            for _ in range(number_of_atoms):
                lines.pop(0)
            ctr = 0
            (x, y, z) = ([], [], [])
            for b in range(number_of_atoms):
                x.append(x_e[b] + x_n[a, b] * h)
                y.append(y_e[b] + y_n[a, b] * h)
                z.append(z_e[b] + z_n[a, b] * h)
                ctr += 1
            (lines, lines_test) = write_lines(file_name_start, number_of_atoms, atomstypes_int, atoms, x, y, z)
            write_input_ams(
                str(a) + "_freq_plus1.run", charge, lines_test, number_of_atoms, file_list_freq, user_input_freq)

            if points == 5:
                for _ in range(number_of_atoms):
                    lines.pop(0)
                ctr = 0
                (x, y, z) = ([], [], [])
                for b in range(number_of_atoms):
                    x.append(x_e[b] + x_n[a, b] * 2 * h)
                    y.append(y_e[b] + y_n[a, b] * 2 * h)
                    z.append(z_e[b] + z_n[a, b] * 2 * h)
                    ctr += 1
                (lines, lines_test) = write_lines(file_name_start, number_of_atoms, atomstypes_int, atoms, x, y, z)
                write_input_ams(
                    str(a) + "_freq_plus2.run", charge, lines_test, number_of_atoms, file_list_freq, user_input_freq)

    for _ in range(number_of_atoms):
        lines.pop(0)
    (lines, lines_test) = write_lines(file_name_start, number_of_atoms, atomstypes_int, atoms, x_e, y_e, z_e)
    write_input_ams(
        "Standard_properties.run", charge, lines_test, number_of_atoms, file_list, user_input_nmr)

    return lines, file_list_freq, file_list, h_list

def read_properties_ams(output_file_name, properties, number_of_atoms):
    motherlist_isotropic = []  # Empty lists to append properties from output files
    motherlist_au = []
    motherlist_megahertz = []
    if "NMR" in properties:  # Find and append properties according to job information
        isotropic = []
        lines_sh = []
        with open(output_file_name, "r") as my_input_sh:
            my_text_sh = my_input_sh.readlines()
            for j, line_sh in enumerate(my_text_sh):
                if "total isotropic shielding" in line_sh:
                    lines_sh.append(line_sh)
            # for line_sh in my_text_sh:  # Check for matching number of atoms
            #     if "NAtoms=    " in line_sh and "NAtoms=    " + str(number_of_atoms) not in line_sh:
            #         os.system('echo "Entered number of atoms not equal to number of atoms in input file."')

        for line_sh in lines_sh:
            isotropic.append(float(line_sh.split()[4]))
            motherlist_isotropic.append(float(line_sh.split()[4]))

        with open(output_file_name, "r") as my_input_sh:  # Check for missing properties
            if "total isotropic shielding" not in my_input_sh.read():
                string = "echo " + "Property not found in file: {0}".format(output_file_name)
                os.system(string)

    elif "J_FC" in properties:
        au = []
        megahertz = []
        lines_fc = []
        start_fc = 0
        with open(output_file_name, "r") as my_input_fc:
            my_text_fc = my_input_fc.readlines()
            for k, line_fc in enumerate(my_text_fc):
                if "Isotropic Fermi Contact Couplings" in line_fc:
                    start_fc += k
            if start_fc > 0:
                for k, line_fc in enumerate(my_text_fc):
                    if k in range(start_fc + 2, start_fc + 2 + number_of_atoms, 1):
                        lines_fc.append(line_fc)
            for line_fc in my_text_fc:
                if "NAtoms=    " in line_fc and "NAtoms=    " + str(number_of_atoms) not in line_fc:
                    os.system('echo "Entered number of atoms not equal to number of atoms in input file."')

        for line_fc in lines_fc:
            word_fc = ""
            for k in line_fc[16:37]:
                if k != " ":
                    word_fc += k
            value_fc = float(word_fc)
            au.append(value_fc)
            motherlist_au.append(value_fc)

        for line_fc in lines_fc:
            word_fc = ""
            for k in line_fc[37:51]:
                if k != " ":
                    word_fc += k
            value_fc = float(word_fc)
            megahertz.append(value_fc)
            motherlist_megahertz.append(value_fc)

        with open(output_file_name, "r") as my_input_fc:
            if "Isotropic Fermi Contact Couplings" not in my_input_fc.read():
                string = "echo " + "Property not found in file: {0}".format(output_file_name)
                os.system(string)

    return motherlist_isotropic, motherlist_au, motherlist_megahertz

def sort_freqs_ams_1(qi, count, list_frequencies, file_list_freq, number_of_freqs):
    motherlist_frequencies = np.zeros((qi, qi * count))
    count_freqs = 0
    for m in range(len(file_list_freq)):
        for j in range(number_of_freqs):
            motherlist_frequencies[j,m] = list_frequencies[count_freqs]
            count_freqs += 1
    return motherlist_frequencies

def sort_freqs_ams(x_n, y_n, z_n, x_s, y_s, z_s, qi, count, number_of_atoms, list_frequencies, frequencies):
    deg_list = []
    i = 0
    while i + 1 < qi:
        if frequencies[i] > frequencies[i + 1] - 10:
            degs = [i, i + 1]
            i += 1
            if i + 1 < qi:
                if frequencies[i] > frequencies[i + 1] - 10:
                    degs.append(i + 1)
                    i += 1
                    if i + 1 < qi:
                        if frequencies[i] > frequencies[i + 1] - 10:
                            degs.append(i + 1)
                            i += 1
                        deg_list.append(degs)
        i += 1
    list_frequencies = np.array(list_frequencies).reshape(qi, qi * count)
    motherlist_frequencies = np.zeros((qi, qi * count))
    x_s = np.array(x_s).reshape(qi * count, qi * number_of_atoms)  # Sort by file
    y_s = np.array(y_s).reshape(qi * count, qi * number_of_atoms)  # Sort by file
    z_s = np.array(z_s).reshape(qi * count, qi * number_of_atoms)  # Sort by file
    for i in range(qi):  # normal mode - file prefix
        for j in range(count):  # plus/minus 1/2 - file suffix
            for deg in deg_list:
                diffs = []
                permutations = []
                for n in deg:  # normal mode in equilibirum file
                    for m in range(2):   # Same or opposite signs
                        for k in deg:  # normal mode in displaced file
                            x = 0
                            y = 0
                            z = 0
                            for l in range(number_of_atoms):
                                if m == 0:
                                    x += np.abs(x_n[k, l] + x_s[j + i * count, l + n * number_of_atoms])
                                    y += np.abs(y_n[k, l] + y_s[j + i * count, l + n * number_of_atoms])
                                    z += np.abs(z_n[k, l] + z_s[j + i * count, l + n * number_of_atoms])
                                elif m == 1:
                                    x += np.abs(x_n[k, l] - x_s[j + i * count, l + n * number_of_atoms])
                                    y += np.abs(y_n[k, l] - y_s[j + i * count, l + n * number_of_atoms])
                                    z += np.abs(z_n[k, l] - z_s[j + i * count, l + n * number_of_atoms])
                            diffs.append(x + y + z)
                diffs = np.array(diffs).reshape(len(deg), len(deg) * 2)
                perms = [a for a in it.permutations(p for p in range(len(deg)))]
                perms = np.array(perms)
                sign_combs = [b for b in it.combinations([0, 1] * len(deg), len(deg))]
                sign_combs = np.array(sign_combs) * len(deg)
                for s in perms:
                    for t in sign_combs:
                        permutations.append(s + t)
                combs = []
                for index_comb in permutations:
                    comb = 0
                    spot = 0
                    for index in index_comb:
                            comb += diffs[spot, index]
                            spot += 1
                    combs.append(comb)
                min_value_index_comb = permutations[combs.index(min(combs))]
                key = []
                for u in min_value_index_comb:
                    while u >= len(deg):
                        u -= len(deg)
                    key.append(u)
                for q in deg:  # Sorted frequency index
                    motherlist_frequencies[i, q + qi * j] = list_frequencies[i, key[q - deg[0]] + deg[0] + qi * j]
            for r in range(qi):
                if motherlist_frequencies[i, r + qi * j] == 0:
                    motherlist_frequencies[i, r + qi * j] = list_frequencies[i, r + qi * j]
    return motherlist_frequencies

def derivatives(
        properties, number_of_atoms, step, qi, motherlist_isotropic, motherlist_au,
        motherlist_megahertz, motherlist_frequencies, elist_isotropic, elist_au,
        elist_megahertz, order, diff_method, points, frequencies, h_list, old_h
        ):
    diff_isotropic = []  # Empty lists to append derivatives
    diff_anisotropy = []
    diff_au = []
    diff_megahertz = []
    diff_frequencies = []
    diff2_isotropic = []
    diff2_anisotropy = []
    diff2_au = []
    diff2_megahertz = []
    bad_list = []
    bad_r2 = []
    if "polynomial" in diff_method:  # Find derivatives using polynomial fit
        if "NMR" in properties:
            motherlist_isotropic = np.array(motherlist_isotropic).reshape(qi, number_of_atoms * 4)
            # motherlist_anisotropy = np.array(motherlist_anisotropy).reshape(qi, number_of_atoms * 4)
            for a in range(0, qi):
                h = h_list[a]
                x = np.array([(-2) * h, (-h), 0, h, 2 * h])  # Define x-axis
                for b in range(0, number_of_atoms):
                    y = np.array(
                        [motherlist_isotropic[a, b], motherlist_isotropic[a, b + number_of_atoms], elist_isotropic[b],
                         motherlist_isotropic[a, b + 2 * number_of_atoms],
                         motherlist_isotropic[a, b + 3 * number_of_atoms]])  # Define y-axis points
                    weights = np.array([1, 2, 4, 2, 1])
                    coefs = poly.polyfit(x, y, order, rcond=None, full=False, w=weights)  # Find polynomial coefficients
                    #f = poly.Polynomial(coefs)  # Define polynomial function
                    #x_new = np.linspace(x[0], x[-1], num=len(x) * 10)  # Define x-axis for plots
                    #y_new = f(x_new)  # Define y-axis points for polynomial
                    #plt.plot(x, y, 'o', x_new, y_new)  # Plot fit
                    dcoefs = poly.polyder(coefs)  # Define derivatives of polynomial coefficients
                    d2coefs = poly.polyder(coefs, 2)  # Evaluate derivatives and convert
                    diff = (poly.polyval(0, dcoefs) / (1.054571817e-14 / (
                            2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27)) ** 0.5 * 0.529177210903)
                    diff2 = (poly.polyval(0, d2coefs) / (1.054571817e-14 / (
                            2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27)) * 0.529177210903 ** 2)
                    diff_isotropic.append(diff)
                    diff2_isotropic.append(diff2)

                    # y = np.array(
                    #     [motherlist_anisotropy[a, b], motherlist_anisotropy[a, b + number_of_atoms],
                    #      elist_anisotropy[b], motherlist_anisotropy[a, b + 2 * number_of_atoms],
                    #      motherlist_anisotropy[a, b + 3 * number_of_atoms]])
                    weights = np.array([1, 2, 4, 2, 1])
                    coefs = poly.polyfit(x, y, order, rcond=None, full=False, w=weights)  # Find polynomial coefficients
                    #f = poly.Polynomial(coefs)
                    #x_new = np.linspace(x[0], x[-1], num=len(x) * 10)
                    #y_new = f(x_new)
                    #plt.plot(x, y, 'o', x_new, y_new)
                    dcoefs = poly.polyder(coefs)
                    d2coefs = poly.polyder(coefs, 2)
                    diff = (poly.polyval(0, dcoefs) / (1.054571817e-14 / (
                            2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27)) ** 0.5 * 0.529177210903
                            )
                    diff2 = (poly.polyval(0, d2coefs) / (1.054571817e-14 / (
                            2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27)) * 0.529177210903 ** 2)
                    diff_anisotropy.append(diff)
                    diff2_anisotropy.append(diff2)

        elif "J_FC" in properties:
            print("Failure")
            motherlist_megahertz = np.array(motherlist_megahertz).reshape(qi, number_of_atoms * 4)
            motherlist_au = np.array(motherlist_au).reshape(qi, number_of_atoms * 4)
            for a in range(0, qi):
                h = step * np.exp((-0.004 * step - 0.0003) * frequencies[a])
                if h < 0.01:
                    h = 0.01
                if old_h == "y":
                    h = step
                x = np.array([(-2) * h, (-h), 0, h, 2 * h])  # Define x-axis
                for b in range(0, number_of_atoms):
                    y = np.array(
                        [motherlist_megahertz[a, b], motherlist_megahertz[a, b + number_of_atoms], elist_megahertz[b],
                         motherlist_megahertz[a, b + 2 * number_of_atoms],
                         motherlist_megahertz[a, b + 3 * number_of_atoms]])
                    weights = np.array([1, 2, 4, 2, 1])
                    coefs = poly.polyfit(x, y, order, rcond=None, full=False, w=weights)  # Find polynomial coefficients
                    """if b == 0:
                        f = poly.Polynomial(coefs)
                        x_new = np.linspace(x[0], x[-1], num=len(x) * 10)
                        y_new = f(x_new)
                        plt.title(f"q = {a}")
                        plt.plot(x, y, 'o', x_new, y_new)
                        r2 = r2_score(y, f(x))
                        if r2 < 0.97:
                            coefs = poly.polyfit(x, y, order + 1, rcond=None, full=False,
                                                 w=weights)  # Find polynomial coefficients
                            x_cut = x[1:4]
                            y_cut = y[1:4]
                            weights_cut = weights[1:4]
                            coefs_cut = poly.polyfit(x_cut, y_cut, order - 1, rcond=None, full=False, w=weights_cut)
                            f_cut = poly.Polynomial(coefs_cut)
                            x_newc = np.linspace(x_cut[0], x_cut[-1], num=len(x_cut) * 10)
                            y_newc = f_cut(x_newc)
                            plt.title(f"q = {a}_cut")
                            plt.plot(x_cut, y_cut, 'o', x_newc, y_newc)
                            r2_cut = r2_score(y_cut, f_cut(x_cut))
                            print(r2_cut)
                            bad_list.append(a)
                            bad_r2.append(r2)
                        else:
                            plt.plot(x, y, 'o', x_new, y_new)
                        plt.show()"""
                    dcoefs = poly.polyder(coefs)
                    d2coefs = poly.polyder(coefs, 2)
                    diff = (poly.polyval(0, dcoefs) / (1.054571817e-14 / (
                            2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27)) ** 0.5 * 0.529177210903)
                    diff2 = (poly.polyval(0, d2coefs) / (1.054571817e-14 / (
                            2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27)) * 0.529177210903 ** 2)
                    """if b == 0 and r2 < 0.97:
                        dcoefs_cut = poly.polyder(coefs_cut)
                        d2coefs_cut = poly.polyder(coefs_cut, 2)
                        diff = (poly.polyval(0, dcoefs_cut) / (1.054571817e-14 / (
                                2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27)) ** 0.5 * 0.529177210903)
                        diff2 = (poly.polyval(0, d2coefs_cut) / (1.054571817e-14 / (
                                2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27)) * 0.529177210903 ** 2)"""
                    diff_megahertz.append(diff)
                    diff2_megahertz.append(diff2)

                    y = np.array(
                        [motherlist_au[a, b], motherlist_au[a, b + number_of_atoms], elist_au[b],
                         motherlist_au[a, b + 2 * number_of_atoms], motherlist_au[a, b + 3 * number_of_atoms]]
                            )
                    weights = np.array([1, 2, 4, 2, 1])
                    coefs = poly.polyfit(x, y, order, rcond=None, full=False, w=weights)  # Find polynomial coefficients
                    # f = poly.Polynomial(coefs)
                    # x_new = np.linspace(x[0], x[-1], num=len(x) * 10)
                    # y_new = f(x_new)
                    # plt.plot(x, y, 'o', x_new, y_new)
                    dcoefs = poly.polyder(coefs)
                    d2coefs = poly.polyder(coefs, 2)
                    diff = (poly.polyval(0, dcoefs) / (1.054571817e-14 / (
                            2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27)) ** 0.5 * 0.529177210903)
                    diff2 = (poly.polyval(0, d2coefs) / (1.054571817e-14 / (
                            2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27)) * 0.529177210903 ** 2)
                    diff_au.append(diff)
                    diff2_au.append(diff2)

        for a in range(0, qi):
            h = h_list[a]
            x = np.array([(-2) * h, (-h), 0, h, 2 * h])  # Define x-axis
            for b in range(0, qi):
                y = np.array(
                    [motherlist_frequencies[a, b], motherlist_frequencies[a, b + qi], frequencies[b],
                     motherlist_frequencies[a, b + 2 * qi], motherlist_frequencies[a, b + 3 * qi]])
                weights = np.array([1, 2, 4, 2, 1])
                coefs = poly.polyfit(x, y, order, rcond=None, full=False, w=weights)  # Find polynomial coefficients
                # f = poly.Polynomial(coefs)
                # x_new = np.linspace(x[0], x[-1], num=len(x) * 10)
                # y_new = f(x_new)
                # plt.figure(b + 1 + qi * a)
                # plt.plot(x, y, 'o', x_new, y_new)
                dcoefs = poly.polyder(coefs)
                diff = poly.polyval(0, dcoefs)
                diff_frequencies.append(diff)

    elif "linear" in diff_method:  # Linear finite difference method
        if "NMR" in properties:
            for a in range(qi):
                h = h_list[a]
                motherlist_isotropic = np.array(motherlist_isotropic).reshape(qi, number_of_atoms * (points - 1))
                motherlist_anisotropy = np.array(motherlist_anisotropy).reshape(qi, number_of_atoms * (points - 1))
                if points == 3:
                    for b in range(number_of_atoms):
                        diff = 0
                        diff += ((motherlist_isotropic[a, b] - motherlist_isotropic[a, b + number_of_atoms]) / (2 * h)
                                 / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27))
                                 ** 0.5 * 0.529177210903)
                        diff_isotropic.append(diff)
                        diff = 0
                        diff += ((motherlist_anisotropy[a, b] - motherlist_anisotropy[a, b + number_of_atoms]) / (2 * h)
                                 / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27))
                                 ** 0.5 * 0.529177210903)
                        diff_anisotropy.append(diff)
                if points == 5:
                    for b in range(number_of_atoms):
                        diff = 0
                        diff += ((motherlist_isotropic[a, b] - 8 * motherlist_isotropic[a, b + number_of_atoms] +
                                  8 * motherlist_isotropic[a, b + 2 * number_of_atoms] -
                                  motherlist_isotropic[a, b + 3 * number_of_atoms]) / (12 * h)
                                 / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27))
                                 ** 0.5 * 0.529177210903)
                        diff_isotropic.append(diff)
                        diff = 0
                        diff += ((motherlist_anisotropy[a, b] - 8 * motherlist_anisotropy[a, b + number_of_atoms] +
                                  8 * motherlist_anisotropy[a, b + 2 * number_of_atoms] -
                                  motherlist_anisotropy[a, b + 3 * number_of_atoms]) / (12 * h)
                                 / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27))
                                 ** 0.5 * 0.529177210903)
                        diff_anisotropy.append(diff)

            for a in range(qi):
                h = step * np.exp((-0.004 * step - 0.0003) * frequencies[a])
                if h < 0.01:
                    h = 0.01
                if old_h == "y":
                    h = step
                for b in range(number_of_atoms):
                    if points == 3:
                        diff2 = 0
                        diff2 += ((motherlist_isotropic[a, b] + motherlist_isotropic[a, b + number_of_atoms]
                                   - 2 * elist_isotropic[b]) /
                                  (h ** 2) / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a]
                                                                 * 1.66053904e-27)) * 0.529177210903 ** 2)
                        diff2_isotropic.append(diff2)
                        diff2 = 0
                        diff2 += ((motherlist_anisotropy[a, b] + motherlist_anisotropy[a, b + number_of_atoms]
                                   - 2 * elist_anisotropy[b]) /
                                  (h ** 2) / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a]
                                                                 * 1.66053904e-27)) * 0.529177210903 ** 2)
                        diff2_anisotropy.append(diff2)
                    if points == 5:
                        diff2 = 0
                        diff2 += ((-motherlist_isotropic[a, b] + 16 * motherlist_isotropic[a, b + number_of_atoms]
                                   - 30 * elist_isotropic[b] + 16 * motherlist_isotropic[a, b + 2 * number_of_atoms]
                                   - motherlist_isotropic[a, b + 3 * number_of_atoms]) /
                                  (12 * h ** 2) / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a]
                                                   * 1.66053904e-27)) * 0.529177210903 ** 2)
                        diff2_isotropic.append(diff2)
                        diff2 = 0
                        diff2 += ((-motherlist_anisotropy[a, b] + 16 * motherlist_anisotropy[a, b + number_of_atoms]
                                   - 30 * elist_anisotropy[b] + 16 * motherlist_anisotropy[a, b + 2 * number_of_atoms]
                                   - motherlist_anisotropy[a, b + 3 * number_of_atoms]) /
                                  (12 * h ** 2) / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a]
                                                                      * 1.66053904e-27)) * 0.529177210903 ** 2)
                        diff2_anisotropy.append(diff2)

        elif "J_FC" in properties:
            for a in range(qi):
                h = step * np.exp((-0.004 * step - 0.0003) * frequencies[a])
                if h < 0.01:
                    h = 0.01
                if old_h == "y":
                    h = step
                motherlist_megahertz = np.array(motherlist_megahertz).reshape(qi, number_of_atoms * (points - 1))
                motherlist_au = np.array(motherlist_au).reshape(qi, number_of_atoms * (points - 1))
                if points == 3:
                    for b in range(number_of_atoms):
                        diff = 0
                        diff += ((motherlist_megahertz[a, b] - motherlist_megahertz[a, b + number_of_atoms]) / (2 * h)
                                 / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27))
                                 ** 0.5 * 0.529177210903)
                        diff_megahertz.append(diff)
                        diff = 0
                        diff += ((motherlist_au[a, b] - motherlist_au[a, b + number_of_atoms]) / (2 * h)
                                 / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27))
                                 ** 0.5 * 0.529177210903)
                        diff_au.append(diff)
                if points == 5:
                    for b in range(number_of_atoms):
                        diff = 0
                        diff += ((motherlist_megahertz[a, b] - 8 * motherlist_megahertz[a, b + number_of_atoms] +
                                  8 * motherlist_megahertz[a, b + 2 * number_of_atoms] -
                                  motherlist_megahertz[a, b + 3 * number_of_atoms]) / (12 * h)
                                 / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27))
                                 ** 0.5 * 0.529177210903)
                        diff_megahertz.append(diff)
                        diff = 0
                        diff += ((motherlist_au[a, b] - 8 * motherlist_au[a, b + number_of_atoms] +
                                  8 * motherlist_au[a, b + 2 * number_of_atoms] -
                                  motherlist_au[a, b + 3 * number_of_atoms]) / (12 * h)
                                 / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a] * 1.66053904e-27))
                                 ** 0.5 * 0.529177210903)
                        diff_au.append(diff)

            for a in range(qi):
                h = step * np.exp((-0.004 * step - 0.0003) * frequencies[a])
                if h < 0.01:
                    h = 0.01
                if old_h == "y":
                    h = step
                for b in range(number_of_atoms):
                    if points == 3:
                        diff2 = 0
                        diff2 += ((motherlist_megahertz[a, b] + motherlist_megahertz[a, b + number_of_atoms]
                                   - 2 * elist_megahertz[b]) /
                                  (h ** 2) / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a]
                                                                 * 1.66053904e-27)) * 0.529177210903 ** 2)
                        diff2_megahertz.append(diff2)
                        diff2 = 0
                        diff2 += ((motherlist_au[a, b] + motherlist_au[a, b + number_of_atoms]
                                   - 2 * elist_au[b]) /
                                  (h ** 2) / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a]
                                                                 * 1.66053904e-27)) * 0.529177210903 ** 2)
                        diff2_au.append(diff2)
                if points == 5:
                        diff2 = 0
                        diff2 += ((-motherlist_megahertz[a, b] + 16 * motherlist_megahertz[a, b + number_of_atoms]
                                   - 30 * elist_megahertz[b] + 16 * motherlist_megahertz[a, b + 2 * number_of_atoms]
                                   - motherlist_megahertz[a, b + 3 * number_of_atoms]) /
                                  (12 * h ** 2) / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a]
                                                   * 1.66053904e-27)) * 0.529177210903 ** 2)
                        diff2_megahertz.append(diff2)
                        diff2 = 0
                        diff2 += ((-motherlist_au[a, b] + 16 * motherlist_au[a, b + number_of_atoms]
                                   - 30 * elist_au[b] + 16 * motherlist_au[a, b + 2 * number_of_atoms]
                                   - motherlist_au[a, b + 3 * number_of_atoms]) /
                                  (12 * h ** 2) / (1.054571817e-14 / (2 * np.pi * 29979245800 * frequencies[a]
                                                                      * 1.66053904e-27)) * 0.529177210903 ** 2)
                        diff2_au.append(diff2)

        for a in range(qi):
            h = step * np.exp((-0.004 * step - 0.0003) * frequencies[a])
            if h < 0.01:
                h = 0.01
            if old_h == "y":
                h = step
            motherlist_frequencies = np.array(motherlist_frequencies).reshape(qi, qi * (points - 1))
            if points == 3:
                for b in range(qi):
                    diff = 0
                    diff += ((motherlist_frequencies[a, b] - motherlist_frequencies[a, b + qi]) / (2 * h))
                    diff_frequencies.append(diff)
            if points == 5:
                for b in range(qi):
                    diff = 0
                    diff += ((motherlist_frequencies[a, b] - 8 * motherlist_frequencies[a, b + qi]
                              + 8 * motherlist_frequencies[a, b + 2 * qi] - motherlist_frequencies[a, b + 3 * qi]) /
                             (12 * h))
                    diff_frequencies.append(diff)

    return (diff_isotropic, diff_anisotropy, diff_au, diff_megahertz, diff_frequencies, diff2_isotropic,
            diff2_anisotropy, diff2_au, diff2_megahertz, bad_list, bad_r2)


def mean_disp(qi, frequencies, k, file_name_start, anharmq):
    q = []  # Empty list to append mean displacements
    q2 = []  # Empty list to append square mean displacements
    start_q = 0
    start_q2 = 0
    if anharmq == "n" or anharmq == "no":
        for r in range(qi):
            q_r = 0
            for s in range(qi):  # Loop normal modes and define displacements
                q_r += (
                        -0.25 * (2 * np.pi * 29979245800 * frequencies[r]) ** (-1.5) * 1.054571817e-34 ** (-0.5)
                        * k[r, s] / 1.66053904e-27 ** 0.5 / 0.529177210903e-10
                )
            q.append(q_r)
        for r in range(qi):  # Loop normal modes and define squared displacements
            q2_r = 1.054571817e-34 / (
                    4 * np.pi * 29979245800 * frequencies[r] * 1.66053904e-27 * 0.529177210903e-10 ** 2)
            q2.append(q2_r)
    elif anharmq == "y" or anharmq == "yes":
        with open(file_name_start.strip("gjf") + "out", "r") as my_input_a:  # Read mean displacements from output
            my_text_a = my_input_a.readlines()
            for a, line_a in enumerate(my_text_a):
                if "Average Normal Coordinates" in line_a:
                    start_q = a + 3
                if "Mean Square Amplitudes of Normal Coordinates" in line_a:
                    start_q2 = a + 3
            for line_a in my_text_a[start_q:start_q + qi]:
                q.append(float(line_a[11:21]))
            for line_a in my_text_a[start_q2:start_q2 + qi]:
                q2.append(float(line_a[10:19]))
    return q, q2

def vib_corr(
        properties, qi, number_of_atoms, q, q2, diff_isotropic,
        diff_anisotropy, diff_au, diff_megahertz, diff2_isotropic, diff2_anisotropy, diff2_au, diff2_megahertz,
        frequencies, temperature, bad_list
        ):
    a_isotropic = []  # Empty lists to append vibrational corrections
    a_anisotropy = []
    a_megahertz = []
    a_au = []
    a1_au = []
    a1_megahertz = []
    a2_au = []
    a2_megahertz = []
    if "NMR" in properties:
        diff_isotropic = np.array(diff_isotropic).reshape(qi, number_of_atoms)
        diff2_isotropic = np.array(diff2_isotropic).reshape(qi, number_of_atoms)
        diff_anisotropy = np.array(diff_anisotropy).reshape(qi, number_of_atoms)
        diff2_anisotropy = np.array(diff2_anisotropy).reshape(qi, number_of_atoms)
        for b in range(number_of_atoms):  # Loop atoms
            a1 = 0
            for r in range(qi):  # Calculate first order contributions for each normal mode
                a1 += 2 * diff_isotropic[r, b] * q[r] / np.tanh(
                    1.054571817e-34 * 29979245800 * frequencies[r] / (2 * 1.38065e-23 * temperature))
            a2 = 0
            for r in range(qi):  # Calculate second order contributions for each normal mode
                a2 += 0.5 * diff2_isotropic[r, b] * q2[r] / np.tanh(
                    1.054571817e-34 * 29979245800 * frequencies[r] / (2 * 1.38065e-23 * temperature))
            a = a1 + a2  # Find total vibrational correction
            a_isotropic.append(a)
            a1 = 0
            for r in range(qi):
                a1 += diff_anisotropy[r, b] * q[r] / np.tanh(
                    1.054571817e-34 * 29979245800 * frequencies[r] / (2 * 1.38065e-23 * temperature))
            a2 = 0
            for r in range(qi):
                a2 += 0.5 * diff2_anisotropy[r, b] * q2[r] / np.tanh(
                    1.054571817e-34 * 29979245800 * frequencies[r] / (2 * 1.38065e-23 * temperature))
            a = a1 + a2
            a_anisotropy.append(a)
    elif "J_FC" in properties:
        diff_au = np.array(diff_au).reshape(qi, number_of_atoms)
        diff2_au = np.array(diff2_au).reshape(qi, number_of_atoms)
        diff_megahertz = np.array(diff_megahertz).reshape(qi, number_of_atoms)
        diff2_megahertz = np.array(diff2_megahertz).reshape(qi, number_of_atoms)
        for b in range(number_of_atoms):
            a1 = 0
            for r in range(qi):
                if r not in bad_list:
                    a1 += diff_megahertz[r, b] * q[r] / np.tanh(
                        1.054571817e-34 * 29979245800 * frequencies[r] / (2 * 1.38065e-23 * temperature))
                a1_megahertz.append(diff_megahertz[r, b] * q[r] / np.tanh(
                    1.054571817e-34 * 29979245800 * frequencies[r] / (2 * 1.38065e-23 * temperature)))
            a2 = 0
            for r in range(qi):
                if r not in bad_list:
                    a2 += 0.5 * diff2_megahertz[r, b] * q2[r] / np.tanh(
                        1.054571817e-34 * 29979245800 * frequencies[r] / (2 * 1.38065e-23 * temperature))
                a2_megahertz.append(0.5 * diff2_megahertz[r, b] * q2[r] / np.tanh(
                    1.054571817e-34 * 29979245800 * frequencies[r] / (2 * 1.38065e-23 * temperature)))
            a = a1 + a2
            a_megahertz.append(a)
            a1 = 0
            for r in range(qi):
                if r not in bad_list:
                    a1 += diff_au[r, b] * q[r] / np.tanh(
                        1.054571817e-34 * 29979245800 * frequencies[r] / (2 * 1.38065e-23 * temperature))
                a1_au.append(diff_au[r, b] * q[r] / np.tanh(
                    1.054571817e-34 * 29979245800 * frequencies[r] / (2 * 1.38065e-23 * temperature)))
            a2 = 0
            for r in range(qi):
                if r not in bad_list:
                    a2 += 0.5 * diff2_au[r, b] * q2[r] / np.tanh(
                        1.054571817e-34 * 29979245800 * frequencies[r] / (2 * 1.38065e-23 * temperature))
                a2_au.append(0.5 * diff2_au[r, b] * q2[r] / np.tanh(
                    1.054571817e-34 * 29979245800 * frequencies[r] / (2 * 1.38065e-23 * temperature)))
            a = a1 + a2
            a_au.append(a)
        a1_au = np.array(a1_au).reshape(number_of_atoms, qi)
        a2_au = np.array(a2_au).reshape(number_of_atoms, qi)
        a1_megahertz = np.array(a1_megahertz).reshape(number_of_atoms, qi)
        a2_megahertz = np.array(a2_megahertz).reshape(number_of_atoms, qi)
    return a_isotropic, a_anisotropy, a_megahertz, a_au, a1_megahertz, a2_megahertz, a1_au, a2_au


def write_results(
        atoms, properties, number_of_atoms, qi, diff_isotropic, diff_anisotropy, diff_au, diff_megahertz,
        cfc, diff2_isotropic, diff2_anisotropy, diff2_au, diff2_megahertz,
        a_isotropic, a_anisotropy, a_megahertz, a_au, elist_isotropic, elist_megahertz, elist_au
        ):
    atom_string = ""
    for i in atoms:
        atom_string = atom_string + "{0:>13}".format(i)
    if "NMR" in properties:
        diff_isotropic = np.array(diff_isotropic).reshape(qi, number_of_atoms)
        diff2_isotropic = np.array(diff2_isotropic).reshape(qi, number_of_atoms)
        diff_anisotropy = np.array(diff_anisotropy).reshape(qi, number_of_atoms)
        diff2_anisotropy = np.array(diff2_anisotropy).reshape(qi, number_of_atoms)
        with open("results_properties.txt", "w") as my_results:
            my_results.write("First derivatives of isotropic:  \n")
            my_results.write(atom_string)
            my_results.write("\n")
            for i in diff_isotropic:
                diff_string = ""
                for a in i:
                    diff_string = diff_string + "{0:13.5f}".format(a)
                my_results.write(diff_string)
                my_results.write("\n")
            my_results.write("Second derivatives of isotropic:  \n")
            my_results.write(atom_string)
            my_results.write("\n")
            for i in diff2_isotropic:
                diff2_string = ""
                for a in i:
                    diff2_string = diff2_string + "{0:13.5f}".format(a)
                my_results.write(diff2_string)
                my_results.write("\n")
            #my_results.write("First derivatives of anisotropy:  \n")
            #my_results.write(atom_string)
            #my_results.write("\n")
            #for i in diff_anisotropy:
             #   diff_string = ""
              #  for a in i:
               #     diff_string = diff_string + "{0:13.5f}".format(a)
               # my_results.write(diff_string)
               # my_results.write("\n")
            #my_results.write("Second derivatives of anisotropy:  \n")
            #my_results.write(atom_string)
            #my_results.write("\n")
            #for i in diff2_anisotropy:
               # diff2_string = ""
                #for a in i:
                   # diff2_string = diff2_string + "{0:13.5f}".format(a)
               # my_results.write(diff2_string)
               # my_results.write("\n")

        with open("results.txt", "w") as my_results:
            my_results.write("Isotropic NMR, ppm:  \n")
            for i in range(number_of_atoms):
                if i + 1 < number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(elist_isotropic[i]) + "\n")
                elif i + 1 == number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(elist_isotropic[i]) + "\n" + "\n")
            my_results.write("Vibrational correction to isotropic NMR, ppm:  \n")
            for i in range(number_of_atoms):
                if i + 1 < number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(a_isotropic[i]) + "\n")
                elif i + 1 == number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(a_isotropic[i]) + "\n" + "\n")
	     my_results.write("Vibrationally Corrected isotropic NMR, ppm: \n")
            for i in range(number_of_atoms):
                if i + 1 < number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(elist_isotropic[i] + a_isotropic[i]) + "\n")
                elif i + 1 == number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(elist_isotropic[i] + a_isotropic[i]) + "\n" + "\n")
            # my_results.write("Anisotropy, ppm:  \n")
            # for i in range(number_of_atoms):
            #     if i + 1 < number_of_atoms:
            #         my_results.write(str(atoms[i]) + ": " + str(elist_anisotropy[i]) + "\n")
            #     elif i + 1 == number_of_atoms:
            #         my_results.write(str(atoms[i]) + ": " + str(elist_anisotropy[i]) + "\n" + "\n")
            # my_results.write("Vibrational correction to anisotropy, ppm:  \n")
            # for i in range(number_of_atoms):
            #     if i + 1 < number_of_atoms:
            #         my_results.write(str(atoms[i]) + ": " + str(a_anisotropy[i]) + "\n")
            #     elif i + 1 == number_of_atoms:
            #         my_results.write(str(atoms[i]) + ": " + str(a_anisotropy[i]) + "\n" + "\n")

    elif "J_FC" in properties:
        diff_au = np.array(diff_au).reshape(qi, number_of_atoms)
        diff2_au = np.array(diff2_au).reshape(qi, number_of_atoms)
        diff_megahertz = np.array(diff_megahertz).reshape(qi, number_of_atoms)
        diff2_megahertz = np.array(diff2_megahertz).reshape(qi, number_of_atoms)
        with open("results_properties.txt", "w") as my_results:
            my_results.write("First derivatives of isotropic Fermi Contact Couplings, a.u.:  \n")
            my_results.write(atom_string)
            my_results.write("\n")
            for i in diff_au:
                diff_string = ""
                for a in i:
                    diff_string = diff_string + "{0:13.5f}".format(a)
                my_results.write(diff_string)
                my_results.write("\n")
            my_results.write("Second derivatives of isotropic Fermi Contact Couplings, a.u.:  \n")
            my_results.write(atom_string)
            my_results.write("\n")
            for i in diff2_au:
                diff2_string = ""
                for a in i:
                    diff2_string = diff2_string + "{0:13.5f}".format(a)
                my_results.write(diff2_string)
                my_results.write("\n")
            my_results.write("First derivatives of isotropic Fermi Contact Couplings, MHz:  \n")
            my_results.write(atom_string)
            my_results.write("\n")
            for i in diff_megahertz:
                diff_string = ""
                for a in i:
                    diff_string = diff_string + "{0:13.5f}".format(a)
                my_results.write(diff_string)
                my_results.write("\n")
            my_results.write("Second derivatives of isotropic Fermi Contact Couplings, MHz:  \n")
            my_results.write(atom_string)
            my_results.write("\n")
            for i in diff2_megahertz:
                diff2_string = ""
                for a in i:
                    diff2_string = diff2_string + "{0:13.5f}".format(a)
                my_results.write(diff2_string)
                my_results.write("\n")

        with open("results.txt", "w") as my_results:
            my_results.write("Isotropic Fermi Contact Couplings, MHz:  \n")
            for i in range(number_of_atoms):
                if i + 1 < number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(elist_megahertz[i]) + "\n")
                elif i + 1 == number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(elist_megahertz[i]) + "\n" + "\n")
            my_results.write("Vibrational correction to isotropic Fermi Contact Couplings, MHz:  \n")
            for i in range(number_of_atoms):
                if i + 1 < number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(a_megahertz[i]) + "\n")
                elif i + 1 == number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(a_megahertz[i]) + "\n" + "\n")
            my_results.write("Isotropic Fermi Contact Couplings, a.u.:  \n")
            for i in range(number_of_atoms):
                if i + 1 < number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(elist_au[i]) + "\n")
                elif i + 1 == number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(elist_au[i]) + "\n" + "\n")
            my_results.write("Vibrational correction to isotropic Fermi Contact Couplings, a.u.:  \n")
            for i in range(number_of_atoms):
                if i + 1 < number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(a_au[i]) + "\n")
                elif i + 1 == number_of_atoms:
                    my_results.write(str(atoms[i]) + ": " + str(a_au[i]) + "\n" + "\n")

    cfc = np.array(cfc).reshape(qi, qi)
    with open("results_freq.txt", "w") as my_results:
        my_results.write("Cubic Force Constants:  \n")
        for i in cfc:
            diff_string = ""
            for a in i:
                diff_string = diff_string + "{0:13.5f}".format(a)
            my_results.write(diff_string)
            my_results.write("\n")


