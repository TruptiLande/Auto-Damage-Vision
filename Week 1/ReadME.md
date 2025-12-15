Genetic Algorithm, Linear Regression, Logistic Regression

/////////////////////////////////////////////////////////
Genetic Algorithm : 
Initialize Population :
REPEAT :
  Evaluate fitness
  Select best individuals
  Create children(crossover)
  Mutate some children
UNTIL solution is good enough

Steps:
1.Representation (Encoding)
  Decide what a solution looks like:
  line = [slope, intercept]
  One individual, two genes : slope, intercept
  GA works only if solutions can be encoded this way.

2. Initial Population
   Start with random lines, GA doesn't require a good starting point

3. Fitness Evaluation :
   Using MSE, SSE etc. It decides who survives

4. Selection : (Survival of the fittest)
   Sort according to errors

5. Crossover (Reproduction)
   Take different genes from different parents

6. Mutation (Random variation)
   Prevents getting stuck in local minima, or population being identical
  
7. New generation replaces old

8. Convergence
