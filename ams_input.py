import numpy as np
from ams_input_functions import read_input_ams
from ams_input_functions import read_inputgeometry_ams
from ams_input_functions import write_input_ams
from ams_input_functions import read_optimized_ams
from ams_input_functions import call_ams
# from ams_input_functions import read_hessian
from ams_input_functions import check_freq_real_eq
from ams_input_functions import check_freq_real
from ams_input_functions import read_and_count_freqs
from ams_input_functions import read_normalcoordinates_ams
from ams_input_functions import read_normalcoordinates_1_ams
from ams_input_functions import read_normalcoordinates_disp_ams
from ams_input_functions import calc_reduced_masses
from ams_input_functions import write_nc_ams
from ams_input_functions import write_loop_ams
from ams_input_functions import write_lines
from ams_input_functions import read_properties_ams
from ams_input_functions import sort_freqs_ams
from ams_input_functions import derivatives
from ams_input_functions import mean_disp
from ams_input_functions import vib_corr
from ams_input_functions import write_results

# cpus = int(os.getenv('CPU'))
# mem = int(os.getenv('MEM'))
cpus = 4

# A: Input variables
(
software, user_input_opt, user_input_freq, user_input_nmr, file_name_start, linear, optimize, file_name_calc_opt_1, file_name_calc_opt_2,
file_name_calc_opt_3, text_line3, text_line4,
atomtypes, number_of_atoms, charge, generators, multiplicity, properties, temperature,
h, functional_p, functional_f, basis_set_p, basis_set_f, diff_method, points, mode, order, anharm,
anharmq, method_geo, basis_geo, method_prop, basis_prop, stepsize, stepnumber, run, old_h) = read_input_ams()

if software == "ams":
# B: Read (and optimize) geometry lines from input file
    (atomstypes_int, lines, atoms, x_e, y_e, z_e) = read_inputgeometry_ams(file_name_start, number_of_atoms)

if optimize == "y" or optimize == "yes":
    print("Optimizing geometry...")
    file_list = write_input_ams("optimization_" + file_name_start.strip("mol") + "run", charge, lines, number_of_atoms,
                                [], user_input_opt)

    call_ams(1, file_list, run)
    (x_e, y_e, z_e) = read_optimized_ams(file_list[0].strip("run") + "out", number_of_atoms)
    file_list.remove("optimization_" + file_name_start.strip("mol") + "run")



# C: Write new input file for normal coordinates
(lines, lines_test) = write_lines(file_name_start, number_of_atoms, atomstypes_int, atoms, x_e, y_e, z_e)

file_list = write_input_ams("freq_" + file_name_start.strip("mol") + "run", charge, lines_test, number_of_atoms, [], user_input_freq)

# D: Run quantum program for optimized geometry and read equilibrium geometry
call_ams(1, file_list, run)

# E: Read normal coordinates, frequencies, reduced masses and irreps from output file
# (hessian) = read_hessian(file_list[0].strip("dal") + "out", atoms)
(number_of_freqs, frequencies) = read_and_count_freqs(file_list[0].strip("run") + "out")
check_freq_real_eq(file_list[0].strip("run") + "out", number_of_freqs)
(x_nm, y_nm, z_nm) = read_normalcoordinates_ams(file_list[0].strip("run") + "out", number_of_atoms)
(x_nm_1, y_nm_1, z_nm_1) = read_normalcoordinates_1_ams(file_list[0].strip("run") + "out", number_of_atoms)
file_list.remove("freq_" + file_name_start.strip("mol") + "run")


if linear == "n" or linear == "no":
    qi = 3 * number_of_atoms - 6
elif linear == "y" or linear == "yes":
    qi = 3 * number_of_atoms - 5
else:
    qi = 0
    print("Missing information about linearity of molecule.")


(reduced_masses) = calc_reduced_masses(number_of_atoms, x_nm, y_nm, z_nm)


# F: Loop normal coordinates
#Reshape, convert and write normal coordinates
x_nm_1 = np.array(x_nm_1).reshape(qi, number_of_atoms)
y_nm_1 = np.array(y_nm_1).reshape(qi, number_of_atoms)
z_nm_1 = np.array(z_nm_1).reshape(qi, number_of_atoms)
x_n = np.zeros((qi, number_of_atoms))
y_n = np.zeros((qi, number_of_atoms))
z_n = np.zeros((qi, number_of_atoms))

for a in range(qi):
    if frequencies[a] > 0:
        for b in range(number_of_atoms):
            x_n[a, b] = x_nm_1[a, b] * (1.0545718e-14 / (2 * 3.14159265359 * 29979245800 * 1.66053904e-27
                                      * frequencies[a] * reduced_masses[b])) ** 0.5
            y_n[a, b] = y_nm_1[a, b] * (1.0545718e-14 / (2 * 3.14159265359 * 29979245800 * 1.66053904e-27
                                      * frequencies[a] * reduced_masses[b])) ** 0.5
            z_n[a, b] = z_nm_1[a, b] * (1.0545718e-14 / (2 * 3.14159265359 * 29979245800 * 1.66053904e-27
                                      * frequencies[a] * reduced_masses[b])) ** 0.5

for a in range(qi):  # Check for imaginary frequencies
    if frequencies[a] < 0:
        del frequencies[a]
        del reduced_masses[a]
        np.delete(x_n, a, 0)
        np.delete(y_n, a, 0)
        np.delete(z_n, a, 0)
        qi -= 1
        inharmonics = "Imaginary harmonic frequency for normal mode: " + str(a + 1)
        print("inharmonics")
        break

print("\nGeometries:")
for i in range(len(atoms)):
    print("{0:} (xyz): {1:< 20.15f}{2:< 20.15f}{3:< 20.15f}".format(atoms[i], x_e[i], y_e[i], z_e[i]))

print("\nFrequencies:")
for i in range(len(frequencies)):
    print("Normal mode {0}: {1}".format(i+1, frequencies[i]))

print("\nNormal modes:")
for i in range(len(frequencies)):
    print("Normal mode {0}".format(i + 1))
    for j in range(len(atoms)):
        print("{0:9}{1:< 20.15f}{2:< 20.15f}{3:< 20.15f}".format(atoms[j], x_n[i, j], y_n[i, j], z_n[i, j]))

write_nc_ams(file_name_start, x_n, y_n, z_n, atoms)

# # Loop normal coordinates writing new input files to then run
print("\nGenerating files for calculations on displaced geometries...")

(lines, file_list_freq, file_list, h_list) = write_loop_ams(number_of_atoms, points, lines, x_e, y_e, z_e, x_n, y_n, z_n, h, atoms, charge,
            file_list, anharm, qi, frequencies, file_name_start, atomstypes_int, user_input_freq, user_input_nmr, h, old_h)

print("Running vib and prop calculations on displaced geometries...")
call_ams(cpus, file_list, run)

if anharm == "n" or anharm == "no":
	call_ams(cpus, file_list_freq, run)

print("Finished with property calculations on displaced geometries!")
print(file_list.extend(file_list))
# Read molecular properties from output files

count = points - 1

(motherlist_isotropic, motherlist_au, motherlist_megahertz) = ([], [], [])
print("\nReading properties at displaced geometries...")
for i in range(qi * count):
    (a, c, d) = read_properties_ams(file_list[i].strip("run") + "out", properties, number_of_atoms)
    motherlist_isotropic.append(a)
    motherlist_au.append(c)
    motherlist_megahertz.append(d)

atom_string = ""
for a in atoms:
    atom_string = atom_string + ("{0:>13}".format(a))

(elist_isotropic, elist_au, elist_megahertz) = read_properties_ams(
          "Standard_properties.out", properties, number_of_atoms)  # Equilibrium properties

if "NMR" in properties:
    print("Isotropic shielding, ppm:")
    print(atom_string)
    for i in motherlist_isotropic:
        nmr_string = ""
        for a in i:
            nmr_string = nmr_string + "{0:13.4f}".format(a)
        print(nmr_string)
    print(atom_string)
elif "J_FC" in properties:
    print("Isotropic Fermi Contact Couplings, a.u.:")
    print(atom_string)
    for i in motherlist_au:
        fc_string = ""
        for a in i:
            fc_string = fc_string + "{0:13.5f}".format(a)
        print(fc_string)
    print("Isotropic Fermi Contact Couplings, MHz:")
    print(atom_string)
    for i in motherlist_megahertz:
        fc_string = ""
        for a in i:
            fc_string = fc_string + "{0:13.5f}".format(a)
        print(fc_string)

print("Finished with vibrational calculations on displaced geometries!")

motherlist_frequencies = np.zeros((qi, qi * count))
list_frequencies = []
k_ijj = []
print("\nReading frequencies at displaced geometries...")

if anharm == "n" or anharm == "no" or anharm == "x":
    (x_s, y_s, z_s, list_frequencies) = read_normalcoordinates_disp_ams(file_list_freq, number_of_freqs, number_of_atoms)
    check_freq_real(file_list_freq, number_of_atoms, number_of_freqs)  # Check for imaginary frequencies
     # Frequency sorting

if mode == "freq":
    motherlist_frequencies = sort_freqs_ams(x_n, y_n, z_n, x_s, y_s, z_s, qi, count, number_of_atoms, list_frequencies, frequencies)

for a in file_list_freq:
    print(a.replace("run", "out"))
for i in motherlist_frequencies:
    frq_string = "disp freq "
    for a in i:
        frq_string = frq_string + "{0:13.5f}".format(a)
    print(frq_string)

# # G: Calculate derivatives
print("\nCalculating numerical first and second derivatives of properties...")
(diff_isotropic, diff_anisotropy, diff_au, diff_megahertz, diff_frequencies, diff2_isotropic, diff2_anisotropy,
    diff2_au, diff2_megahertz, bad_list, bad_r2) = \
    derivatives(properties, number_of_atoms, h, qi, motherlist_isotropic, motherlist_au,
                motherlist_megahertz, motherlist_frequencies, elist_isotropic, elist_au,
                elist_megahertz, order, diff_method, points, frequencies, h_list, old_h)

if "NMR" in properties:
    print("\nFirst derivatives of shieldings:")
    for i in range(len(atoms)):
        print("{0}:".format(atoms[i]))
        for j in range(len(frequencies)):
            print(diff_isotropic[i + (j * len(atoms))])
    print("\nSecond derivatives of shieldings:")
    for i in range(len(atoms)):
        print("{0}:".format(atoms[i]))
        for j in range(len(frequencies)):
            print(diff2_isotropic[i + (j * len(atoms))])

elif "J_FC" in properties:
    print("\nFirst derivatives of FC:")
    for i in range(len(atoms)):
        print("{0}:".format(atoms[i]))
        for j in range(len(frequencies)):
            print(diff_megahertz[i + (j * len(atoms))])
    print("\nSecond derivatives of FC:")
    for i in range(len(atoms)):
        print("{0}:".format(atoms[i]))
        for j in range(len(frequencies)):
            print(diff2_megahertz[i + (j * len(atoms))])

print("\nCalculating cubic force constants...")
if mode == "fc":
    diff_frequencies = 0.5 * k_ijj
cfc = 2 * np.array(diff_frequencies).reshape(qi, qi)

print("\nCubic force constants:")
for i in range(qi):
    for j in range(qi):
        print("k_{0}{1}{1}: {2}".format(i+1, j+1, cfc[i,j]))


# H: Calculate vibrational corrections
k = cfc * 29979245800 * 6.62606896e-34  # Cubic force constants

print("\nCalculating mean and mean square displacements...")
(q, q2) = mean_disp(qi, frequencies, k, "anh_" + file_name_start, anharmq)  # Mean displacements
print("\nMean displacements:")
for i in range(len(frequencies)):
    print("Normal mode {0}: {1}".format(i+1, q[i]))
print("\nMean square displacements:")
for i in range(len(frequencies)):
    print("Normal mode {0}: {1}".format(i+1, q2[i]))

(a_isotropic, a_anisotropy, a_megahertz, a_au, a1_megahertz, a2_megahertz, a1_au, a2_au) = \
    vib_corr(properties, qi, number_of_atoms, q, q2, diff_isotropic, diff_anisotropy, diff_au, diff_megahertz,
             diff2_isotropic, diff2_anisotropy, diff2_au, diff2_megahertz, frequencies, temperature, bad_list)
if "NMR" in properties:
    print("\nCalculating vibrational correction to shieldings...")
    print("\nVibrational Correction to shielding, isotropic:")
    for j in range(len(atoms)):
        print("{0}: {1}".format(atoms[j], a_isotropic[j]))
    # print("\nVibrational Correction to shielding, anisotropy:")
    # for j in range(len(atoms)):
    #     print("{0}: {1}".format(atoms[j], a_anisotropy[j]))
elif "J_FC" in properties:
    print("\nCalculating vibrational correction to Fermi Contact term...")
    # print("\nContributions to Vibrational Correction to Fermi Contact term of core atom, a.u.:")
    print("\nFirst term:")
    for j in range(qi):
        print("Mode {0}: {1}".format(j+1, ai_au[0,j]))
    print("\nSecond term:")
    for j in range(qi):
        print("Mode {0}: {1}".format(j+1, a2_au[0,j]))
    print("\nSums:")
    print("First term: {0}".format(np.sum(a1_au[0])))
    print("Second term: {0}".format(np.sum(a2_au[0])))
    print("\nContributions to Vibrational Correction to Fermi Contact term of core atom, MHz:")
    print("\nFirst term:")
    for j in range(qi):
        print("Mode {0}: {1}".format(j+1, a1_megahertz[0, j]))
    print("\nSecond term:")
    for j in range(qi):
        print("Mode {0}: {1}".format(j+1, a2_megahertz[0, j]))
    print("\nSums:")
    print("First term: {0}".format(np.sum(a1_megahertz[0])))
    print("Second term: {0}".format(np.sum(a2_megahertz[0])))
    print("\nModes omitted due to bad polynomial fit:")
    for r in range(len(bad_list)):
        print("Mode {0}: R^2 = {0}".format(bad_list[r]))
    print("\nVibrational Correction to Fermi Contact term, a.u.:")
    for j in range(len(atoms)):
        print("{0}: {1}".format(atoms[j], a_au[j]))
    print("\nVibrational Correction to Fermi Contact term, MHz:")
    for j in range(len(atoms)):
        print("{0}: {1}".format(atoms[j], a_megahertz[j]))

# I: Write results and terminate
write_results(atoms, properties, number_of_atoms, qi, diff_isotropic, diff_anisotropy, diff_au, diff_megahertz,
              cfc, diff2_isotropic, diff2_anisotropy, diff2_au, diff2_megahertz, a_isotropic,
              a_anisotropy, a_megahertz, a_au, elist_isotropic, elist_megahertz, elist_au)

print("Processes terminated normally.")


