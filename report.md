<div>
    <p>
        Donovan Henry
    </p>
    <p>
        dphenry
    </p>
    <p>
        1799761
    </p>
</div>

<center>
    <h1>
        Implementation Details
    </h1>
    <h2>
        Default Hillclimb
    </h2>
    <p>
        For default hillclimb, I start by generating a path from the start vertex, i.e. vertex 0, back to itself, and I ensure that every vertex is visited once. Initially, I was just going to repeat this process in a depth-first search, but I decided that I would actually start with a path that visited every node exactly once. In mathematics, we call this a hamiltonian cycle. After we have a hamiltonian cycle, we make pairwise swaps to get the neighboring states. Then, among these neighboring states, we find the most optimal one, and if it is any better than our current state, we set our state to this neighboring state, and continue the process of finding neighbors again.
    </p>
    <h2>
        Random restart Hillclimb
    </h2>
    <p>
        For random restart hillclimbing, I randomly generate a hamiltonian cycle. This function will surely execute very slowly for large and very unconnected graphs because of the random method described in the previous sentence. Anyway, after generating that random state, I perform the same process as the default hillclimb. The only difference is that I iterate for (some aribtrary number) N number of times. 
    </p>
    <h2>
        Stochastic Hillclimb
    </h2>
    <p>
        I generate a hamiltonian cycle from 0 to 0 the same way I did in the default hillclimb. Then, I generate the neighbors of the current state, and stochastically select a neighbor by some probability dependent on the heuristic of each neighboring state.
    </p>
    <h2>
        Simulated Annealing
    </h2>
    <p>
        I maintain a priority of which jobs the machines should be focusing on. Then, with that priority, I generate a schedule for each machine, and with that schedule, I determine the heuristic of that particular schedule. Next, I do a pairwise swap of the current state to get a neighoring state, and if the cost of that state is less than the cost of the current state, i.e., has a better heuristic, than I set the current state to the neighboring state. If not, then using a sigmoid, I probabilistically set the current state to the neighboring state, dependent on the temperature and the differential in heuristics. After the temperature of the system hits zero, I perform a stagnation check. For each time that the neighbor's cost is the same as the current cost, I increment the stagnation count, and once it hits some arbitrary number, I exit the function.
    </p>
    <h2>
        Genetic Algorithm
    </h2>
    <p>
        I generate k random priorities for the machines. Then, I base my code off lecture 7 slide 33. I create an array for the newly reproduced offspring. For N number of times (population size), I stochastically select two priorities based off of how fit their heuristics are and perform a cross-over with them, to merge what hopefully makes them good. Then, with a 75% chance of occurring, I perform a pairwise swap to mimic genetic mutation, thus preventing stagnation/convergence to a particular family. At the end of execution, I loop through all states in the final population, select the one with the most optimal heuristic, and print out the schedule of it and it's final cost.
    </p>
    <h2>
        Map Coloring
    </h2>
    <p>
        I create a list of colors (one for each vertex) and color each vertex. If I can validly color a vertex a certain color, I do so. If not, then that mins I need at least one more color, and I redo the coloring process. The number of times iterated is the number of colors required to fill in the graph.
    </p>
</center>

<center>
    <h1>
        Comparing results of each case
    </h1>
    <h2>
        Case 1: Travelling Salesperson
    </h2>
    <p>
        Time to execute all versions of hill-climbing: 0.12 seconds
        Number of executions: 8
        => Average execution time: 0.015 seconds
    </p>
    <p>
        Time to execute all versions of random-restart hill-climbing: 36.2 seconds
        Number of executions: 8
        => Average execution time: 4.525 seconds
    </p>
    <p>
        Time to execute all versions of hill-climbing: 0.206 seconds
        Number of executions: 8
        => Average execution time: 0.026 seconds
    </p>
    <h2>
        Case 2: Job-Shop Scheduler
    </h2>
    <p>
        Time to execute all versions of search annealing: 6.125 seconds
        Number of executions: 7
        => Average execution time: 0.875 seconds
    </p>
    <p>
        Time to execute all versions of genetic algorithm: 3.949 seconds
        Number of executions: 7
        => Average execution time: 0.564 seconds
    </p>
    <h2>
        Case 3: Performance check of TSP
    </h2>
    <p>
        Time to execute all versions of hill-climbing: 0.259 seconds
        Number of executions: 5
        => Average execution time: 0.052 seconds
    </p>
    <p>
        Time to execute all versions of random-restart hill-climbing: 23.697 seconds
        Number of executions: 5
        => Average execution time: 4.739 seconds
    </p>
    <p>
        Time to execute all versions of hill-climbing: 0.341 seconds
        Number of executions: 5
        => Average execution time: 0.068 seconds
    </p>
</center>

