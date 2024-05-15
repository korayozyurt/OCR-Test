import numpy as np
import random as rnd

iterasyon = 50
crosover_rate = 0.8
pop_size = 6
gen_size = 4
fitness_values = []
epok = 0     #gen) belirli bir datanın verildiği belirli bir an. belirli olayların meydana geldiği yer. Çağ, devir.

def main_function(cromosomes):
    return (cromosomes[0] + cromosomes[1] * 2 + cromosomes[2] * 3 + cromosomes[3] * 4) - 30

def fitness_function(cromosomes):
    return 1 / (1 + abs(main_function(cromosomes)))

def create_chromosome():
    return [rnd.randint(0,30) for x in range(0,gen_size)]

def create_initial_pop():
    #return [create_chromosome() for x in range(0, pop_size)]
    return [[3,3,13,13],[5,10,18,28],[29,21,18,8],[9,22,2,21],[13,21,16,2],[18,5,16,28]]

def probability(fitness_values):
    P = []
    total = sum(fitness_values)
    for f in fitness_values:
        P.append(f / total)
    return P

def crossover(P1, P2):
    O1 = []
    O2 = []

    c = rnd.randint(0, gen_size -1)
    O1[:c] = P2[:c]
    O1[c:] = P1[c:]

    O2[:c]=P1[:c]
    O2[c:]=P2[c:]
    return O1,O2

def mutation(mut):
    temp = mut[:]
    gen = rnd.randint(0,30)
    index = rnd.randint(0, gen_size-1)
    temp[index] = gen
    return temp;

def printPopulationAndFitnessFunc():
    print("popülasyon:")
    print(population)
    print("popülasyonun ana fonksiyon değerleri")
    for x in population:
        print(main_function(x), end=", ")
        print(fitness_values[-1], end="\n")

population = create_initial_pop()
for i in population:
    fitness_values.append(fitness_function(i))
printPopulationAndFitnessFunc()
print("İterasyon boyunca algoritma çalıştırılıyor")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
while(epok < iterasyon):
    # SELECTION SECTION
    # Öncelikle seçilme olasılığını hesaplıcaz. P[i] = fitness/total fitness seçilme olasılığını belirler.
    P = probability(fitness_values)
    print("seçilme olasılıkları:" , P)
    c = np.cumsum(P)     #arr.push( i > 0 ? c[i] + c[i-1] : c[i]   )

    # pop_size kadar değerimiz olduğu için ve kümülatif sum length aynı olduğu için len(c) kadar iterasyon yapabiliriz.
    # rastgele bir sayı belirliyoruz 0-1 arasında rnd.random() ile. Bir genin seçilme olasılığı ne kadar büyükse kümülatifte de değer artışı o kadar çok olacağı için
    # eğer bu random sayıdan büyükse seçilecek.

    print("test fitness func:" , fitness_values)
    rulet_parents = []
    for i in range(0, len(c)):
        r = rnd.random()
        for j in range(0, len(c)):
            if c[j] > r:
                rulet_parents.append(j)
                break
    print("rulet parents: ", rulet_parents)

    #Cross Over SECTION
    crosover_parents = []
    k = 0
    while(k < pop_size):
        r = rnd.random()
        if(r < crosover_rate):
            if(rulet_parents[k] not in crosover_parents):
                crosover_parents.append(rulet_parents[k])
        k=k+1
    print("crosover parents: ", crosover_parents)

    if(len(crosover_parents) > 1):
        for i in range(0, len(crosover_parents)):
            for j in range(i+1, len(crosover_parents)):
                print("test population:", population)
                print("crossover:", crosover_parents)
                print("i: ",i)
                print("j: ", j)
                o1,o2 = crossover(population[crosover_parents[i]], population[crosover_parents[j]])
                population.append(o1)
                population.append(o2)
                fitness_values.append(fitness_function(o1))
                fitness_values.append(fitness_function(o2))
    else:
        print("Çaprazlamadan birey gelmedi", " ", epok)


    # MUTATION SECTION
    for i in range(0,50):
        mut = mutation(population[rnd.randint(0, len(population)-1)])
        population.append(mut)
        fitness_values.append(fitness_function(mut))


    # Burada fitness değeri küçük olanları eleyip popülasyonu tekrar yapıyoruz.

    # SURVIVOR SELECTON
    zip_list = zip(fitness_values, population)
    sort_list = sorted(zip_list, reverse=True)

    e= len(population)
    while(e > pop_size):
        sort_list.pop()
        e = e-1

    population = []
    fitness_values = []
    for i,j in list(sort_list):
        population.append(j)
        fitness_values.append(i)
    printPopulationAndFitnessFunc()

    epok = epok +1
