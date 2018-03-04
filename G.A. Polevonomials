from math import sin, pi, cos
from random import choice, shuffle, sample
from itertools import permutations
from numpy.random import choice as choose
import re

# ╭---------------------------------------------------------------╮
# ┃                                                               ┃
# ┃ Mathematical Function approximation with n degree polynomials ┃
# ┃                                                               ┃
# ┃        by Genetic Algorithm       --     Cestpasgrave         ┃
# ┃                                                               ┃
# ╰---------------------------------------------------------------╯
#
#
# Settings here :

func = 'sin(x)'     # Write here in a string the function you want to approximate
                    # If you want to use something different from sin(x), make sure
                    # you import the real function (e.g: from math import cos)
                    # ALTERNATIVE :---------
                    # If you want to find a polynomial passing threw some defined points
                    # instead of a function, just write it as a dictionary in func.
                    # something in the shape {x1:y1,x2:y2,x3:y3,x4:y4}
                    # And you can then put '0' or anything for gap and res.
                    # -----------------------------------------------------
gap = [0, pi / 2]   # Gap is the interval inside which you want the function to be approximated
res = 50            # res is the resolution of the test points used to evaluate the quality of the approx
                    # 50 means that 50 evenly spaced points inside the gap interval will be used
                    # to test the difference between the real function and the one produced by the G.A.
size = 1000         # Size of the population used during evolution.
                    # It's the size of initial population and of population after selection
                    # It will significantly grow during reproduction
generations = 300   # Number of generations iterated
deg_min = 9         # Minimal possible degree in initial population
deg_max = 17        # Maximal possible degree in initial population


# Other degrees can appear during evolution,
# resulting from mutation or reproduction
# deg_min and deg_max must be positive integers, > 0.
# deg_max must be under 100. (and deg_min = 1 will produce a constant, not very useful)

#  With these settings, I managed to get the following polynomial for sin(x) on [0,pi/2]
#  9.9999*10^-4 + 9.5999919*10^-1*x + 5.250005035077*10^-2*x^2 - 0.172877*10^0*x^3
#  - 6.*10^-6*x^4-0.*10^-6*x^5-9.*10^-19*x^6 + 5.5*10^-5*x^7 + 7.995*10^-4*x^8
# It's not as good as Taylor/Maclaurin, but it's looking quite nice on Desmos :
# www.desmos.com
# You can copy the next line into Desmos and compare it to another line containing sin(x):
# 9.9999*10^{-4}+9.5999919*10^{-1}*x+5.250005035077*10^{-2}*x^2-0.172877*10^0*x^3-6.*10^{-6}*x^4-0.*10^{-6}*x^5-9.*10^{-19}*x^6+5.5*10^{-5}*x^7+7.995*10^{-4}*x^8


def test_values(func, gap, res):

    # Testing values used to evaluate the polynomial
    # On the given interval (gap)
    # ref is a dictionary,
    # his keys are the test values,
    # his values are all actual func(x)

    test = []
    if type(func) == str:
        a, b = gap[0], gap[1]
        for i in range(res + 1):
            test.append((abs(b - a) / res) * i)
        ref = {i: j for i, j in zip(test, [eval(func) for x in test])}
    else:
        ref = func
    return ref

def populate(pop, code, size, deg_l, deg_min, deg_max):

    # The population is made of genetic codes, each of which is able to code for  an 'n' degree polynomial
    # The 'pop' list is made of coding strings, all made of 'ATGC' letters
    # ex : 'TCACGTATCCACGCGTCAAAGTGGTTCTTGTTCAGGGCGCGGCATAGTCGTATCCGCCATCACACGGGCACGAAAACTGATCCGCAATGTACAAAA'
    # deg_l is a parallel list containing the degrees for each polynomial at the correspondng indexes

    for i in range(size):
        poly = '';
        stop = 0
        deg = choice(range(deg_min, deg_max + 1))
        deg_l.append(deg)
        while stop < deg + 1:
            temp = choice(list(code.keys()))
            if code[temp] == 'stop':
                if len(poly) == 0:
                    continue
                if code[poly[-3:]] == 'stop':
                    continue
                stop += 1
            poly += temp
        poly = poly[:-3]
        pop.append(poly)
    return pop, deg_l

def express(pop, code):

    # This function is where the genetic 'ATGC' code is transformed into a polynomial that can be 'eval'ed
    # by the python eval() function. The first three letters of the gene are coding for special elements
    # such as the general sign, the power of 10's sign and the eventual +10 for this power
    # The expressed polynomial will look like :
    # (-)d.(dd...)e(-)(1)d*x**(d)d where d is a digit and parenthesis mean optional content.

    e_pop = []  # e_pop for expressed population
                # e_pol for expressed polynomial
                # e_mon for expressed monomial
                # tri is the current codon (3 letters basic unity of the genetic code)

    for i in range(len(pop)):

        e_pol = [];
        pol = [];
        pow = 0;
        e_mon = '';
        mon = ''
        gene = pop[i]

        for j in range(len(gene) // 3):
            tri = gene[j * 3:j * 3 + 3]

            if code[tri] == 'stop' or j == len(gene) // 3 - 1:
                if j == len(gene) // 3 - 1 and code[tri] != 'stop':
                    e_mon += str(code[tri])
                    mon += tri
                if len(mon) < 4:
                    e_mon = '0.0'
                    mon = choice(list(code.keys())[:-4])
                else:
                    head = mon[:3]  # head is the monomial header, containing info about sign and power
                    sign = '-' if head[0] in 'AT' else ''
                    sign2 = '-' if head[1] in 'AT' else ''
                    pow10 = '1' if head[2] == 'A'  else ''
                    e_mon = '{}{}.{}e{}{}{}*x**{}'.format(sign, e_mon[1], e_mon[2:-1], sign2, pow10, e_mon[-1:], pow)
                e_pol.append(e_mon)
                pol.append(mon)
                if code[tri] == 'stop':
                    pol.append(tri)
                e_mon = ''; mon = ''; pow += 1

            else:
                mon += tri
                e_mon += str(code[tri])

        pol = ''.join(pol)
        pop[i] = pol
        e_pol = '+'.join(e_pol)
        e_pop.append(e_pol)

    return pop, e_pop

def evaluate(pop, code, func, gap, res):

    pop, e_pop = express(pop, code)

    # Evaluate() func calculates the result given by each polynomial in the population
    # whis x being each key from the ref dictionary, then calcuates the absolute difference
    # between each result and the actual func(x) result (ref dic values) -> 'distances'
    # Adding all the distances will give us an evaluation value
    # for this polynomial. The bigger it is, the worst the polynomial is at
    # approximating func(x). We'll use it for selection.

    ref = test_values(func, gap, res)

    distances = []

    for e_pol in e_pop:
        diff = 0
        for i in range(len(ref.keys())):
            x = list(ref.keys())[i]
            diff += abs(eval(e_pol) - list(ref.values())[i])
        distances.append(diff)

    # Next it rebuilds the degree list, mutation may have modified the deg number of some polynomials.

    deg_l = []

    for p in range(len(pop)):
        pol = pop[p]
        stop = 0
        if code[pol[:3]] == 'stop':
            stop -= 1
        if code[pol[-3:]] == 'stop':
            stop -= 1
        for i in range(len(pol) // 3):
            if code[pol[3 * i:3 * i + 3]] == 'stop':
                stop += 1
        deg_l.append(stop)

    return deg_l, distances

def fitness(pop, deg_l, distances, dis=10):

    # Here population ('pop'), differences ('distances')
    # and deg_l are ordered accordingly,
    # from the best to the worst polynomials
    # And a fitness list with corresponding indexes is produced.

    rang = [*range(len(pop))]
    rang = sorted(rang, key=lambda x: distances[rang.index(x)])
    o_pop = [];
    o_deg = [];
    o_dist = []
    for i in rang:
        o_pop.append(pop[i])
        o_deg.append(deg_l[i])
        o_dist.append(distances[i])
    pop = o_pop
    deg_l = o_deg
    distances = o_dist

    # dis is the ratio we want between highest and lowest fitness attributed.
    # All fitness values will then be evenly ranged between the two extrema.

    fit = [(i - 1) * (dis - 1) / (len(pop) - 1) + 1 for i in range(len(pop), 0, -1)]

    return pop, distances, fit, deg_l

def reproduce(female_pop, male_pop, male_deg_l, male_fit, sort_by_degree):

    # Female and male is not like biological females and males.
    # It's just a solution allowing the algo to use different
    # populations for mating them. The female population is the
    # one that will be looped in, each individual from this pop
    # will have two children. The males are chosen in the male pop
    # that can be the same pop, they'll be chosen randomly with some rules.
    # Reproduction is made between genes coding for polynomials with the same
    # degree. It consists in a random number (1, 2 or 3) of crossovers.
    # https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm

    children = []

    for i in range(len(female_pop)):   # Each individual from population will produce 2 children
                                    # by crossing his gene with another parent producing the
                                    # same degree polynomial, chosen randomly with a weighted
                                    # probability based on the fitness.

        if sort_by_degree == 1:                                       # Case of intra-degrees reproduction
            indx = [j for j, x in enumerate(male_deg_l) if x == male_deg_l[i]]  # indx is the index of parents
                                                                      # producing the same degree polynomial
            fitx = [male_fit[k] for k in indx]                        # fitx is the corresponding fitness
            fitx = [m / sum(fitx) for m in fitx]                      # list to these parents
            d = male_pop[choose(indx, p=fitx)][:]                     # d is the male (father code)
        else:
            male_fit = [m / sum(male_fit) for m in male_fit]
            d = male_pop[choose([*range(len(male_pop))], p=male_fit)] # Case of inter-degrees reproduction
        c = female_pop[i][:]                                          # c is the female (mother) code

        cross = choice([1, 2, 3])               # cross is the randomly chosen number of successive
        for cr in range(cross):                 # crossovers operation
            cut = choice(range(1, min(len(c), len(d))))
            cut = choice([-cut, cut])
            c = c[:cut] + d[cut:]
            d = d[:cut] + c[cut:]
        children.extend([c, d])     # The resulting two children are added to the population
                                    # Parent are not suppressed, at this point the population
                                    # triples. It will be reduced in selection()
    female_pop.extend(children)
    return female_pop

def mutate(pop, code, spreading, mut_rate=0.005):

    # All different types of random mutations :
    # https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm
    mut_rate = mut_rate + 0.008/(1+spreading)   # spreading is the difference between the best and
                                                # the worst distances. The mutation rate will increase
                                                # as spreading becomes smaller. This helps the population
                                                # to keep on evolving and finding new solutions when
                                                # it becomes too much homogenic.
    def flip(gene):
        loc = choice([*range(len(gene))])
        gene = gene[:loc] + choice('ATGC') + gene[loc + 1:]
        return gene

    def swap(gene):
        loc_a = choice([*range(len(gene))])
        loc_b = choice([*range(loc_a, len(gene))])
        gene = gene[:loc_a] + gene[loc_b] + gene[loc_a + 1:loc_b] + gene[loc_a] + gene[loc_b + 1:]
        return gene

    def scramble(gene):
        set = choice([*range(len(gene) // 10)])
        loc = choice([*range(len(gene) - set)])
        rep = gene[loc:loc + set]
        gene = gene[:loc] + ''.join(sample(rep, k=len(rep))) + gene[loc + set:]
        return gene

    def inverse(gene):
        set = choice([*range(2, 11)])
        loc = choice([*range(len(gene) - set)])
        rep = gene[loc:loc + set]
        gene = gene[:loc] + rep[::-1] + gene[loc + set:]
        return gene

    def insert(gene):
        set = 3 * choice([*range(1, 5)])
        loc = choice([*range(len(gene))])
        ins = ''.join(choose(['A', 'T', 'G', 'C'], size=set))
        gene = gene[:loc] + ins + gene[loc:]
        return gene

    def delete(gene):
        set = 3 * choice([*range(1, 5)])
        loc = choice([*range(len(gene) - set)])
        gene = gene[:loc] + gene[loc + set:]
        return gene

    for i in range(len(pop)):   # mutations are possibly applied to each gene of the population.
                                # The default rate is 0.005, 5 on 1000 bases probabiliy to occur
                                # then the kind of mutation is randomly chosen
        gene = pop[i]
        mut = sum((choose([0, 1], size=len(pop[i]), p=[1 - mut_rate, mut_rate])))
        for i in range(mut):
            gene = choose([flip(gene), swap(gene), scramble(gene), inverse(gene), insert(gene), delete(gene)])
        if code[gene[:3]] == 'stop':
            gene = gene[3:]
        if code[gene[-3:]] == 'stop':
            gene = gene[:-3]
        pop[i] = gene

    return pop

def selection(pop, fit, deg_l, size):

    # Selection is very simple.
    # The 'n' elements of the population with the best fittness are selected
    # where 'n' is the original size of the population.
    # fit and deg_l lists are sliced accordingly

    pop = pop[:size]
    fit = fit[:size]
    deg_l = deg_l[:size]

    return pop, fit, deg_l

def print_it(pop, size, code, res, distances, deg_l):
    print('Best gene ------> ', pop[0], sep='')
    print('Codes for ------> ', express([pop[0]], code)[1][0],sep='')
    print('----------------- (',deg_l[0],' degrees polynomial)',sep='')
    print('Shortened form -> ',shorten_it(express([pop[0]],code)[1][0])+' ')
    print('')
    best = distances[0]
    print('Best distance from reference ---> {} --> ({:.4f}... per point)'.format(best,best/res),sep = '')
    average = sum(distances[:size])/size
    print('Average ------------------------>',average)
    worst = distances[size-1]
    print('Worst -------------------------->',worst)
    print('')
    print('    -- Population --')
    print('    deg')
    degrees = set(deg_l)
    for i in degrees:
        d = deg_l.count(i)
        print('    {:2d} '.format(i) + int(d*50/len(deg_l))*'█', d)
    spreading = worst - best
    return spreading

def shorten_it(epol):
    epol = epol.replace('+-', '-')
    epol = epol.replace('+', ' + ')
    epol = epol.replace('*x', 'x')
    epol = epol.replace('**', '^')
    epol = epol.replace('x^0 ',' ')
    epol = epol.replace('x^1 ', 'x ')
    epol = epol.replace('+ 0.0 ', ' ')
    pattern = '([0-9].[0-9]{3})[0-9]*'
    epol = re.sub(pattern, r'\1', epol)
    pattern = '([0-9])(-)([0-9])'
    epol = re.sub(pattern, r'\1 \2 \3', epol)
    return epol

def evolve(func, gap, res, size, generations, deg_min, deg_max):

    # ----------Genetic code :--------------------------------------------------
    # There are 64 possible triplets with 'ATGC' bases (codons -> cod)
    # Each of the 60 first are coding for single numbers (0 - 9).
    # There are 6 codons for each number.
    # Then the last 4 codons are 'stop' codons, they code for the 'stop' string.
    # The expression of codons are a_a for aminoacids (numbers or 'stop' strings.
    # The genetic code is intentionally fixed, so that genes can be stored and keep
    # their meaning.

    cod = sorted(list({''.join(i) for i in permutations('AAATTTCCCGGG', 3)}))
    a_a = [i for i in range(10) for j in range(6)] + ['stop'] * 4
    code = {i: j for i, j in zip(cod, a_a)}

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Here comes the evolution process
    # First : initial population is generated
    deg_l = []
    pop = []

    # Generating population
    pop, deg_l = populate(pop, code, size, deg_l, deg_min, deg_max)

    # Calculating the fitness
    deg_l, distances = evaluate(pop, code, func, gap, res)
    pop, distances, fit, deg_l = fitness(pop, deg_l, distances)
    best = pop[:2]          # keeping the 5 best genes to save them from mutation
                            # cause somtimes a new solution is unique and may
                            # disappear within one generation.

    spreading = print_it(pop, size, code, res, distances, deg_l)

    # Then successive generations will happen with the same pattern :

    for g in range(1, generations + 1):
        # Reproduction (female pop and male pop are the same here)
        # The default reproduction enables mating between same degree only
        # But it seems more efficient when the different degrees can mate

        temp = pop[:]
        pop = reproduce(pop, pop, deg_l, fit, 0) # The 0 at the end is for
                                                 # sort_by_degree = None
        # Mutation
        pop = mutate(pop, code, spreading)
        pop.extend(best)

        # Addition of new fresh random individuals (they reproduce 1 gen before mixing)
        new = [] ; new_deg = []
        new, deg_l = populate(new, code, size // 4, new_deg, deg_min, deg_max)
        # The reproduction is made for each of the new individuals, but males are
        # chosen from the already present population.
        new = reproduce(new, temp, deg_l, fit,0)
        pop.extend(new)

        # Calculating the fitness (and sorting lists)
        deg_l, distances = evaluate(pop, code, func, gap, res)
        pop, distances, fit, deg_l = fitness(pop, deg_l, distances)

        # Selection
        pop, fit, deg_l = selection(pop, fit, deg_l, size)
        best = pop[:2]      # keeping the 5 best genes to save them from mutation
                            # cause somtimes a new solution is unique and may
                            # disappear within one generation.

        print('\n------------------- GENERATION ', g)
        print('--------------------------------- ')
        spreading = print_it(pop, size, code, res, distances, deg_l)

evolve(func, gap, res, size, generations, deg_min, deg_max)
