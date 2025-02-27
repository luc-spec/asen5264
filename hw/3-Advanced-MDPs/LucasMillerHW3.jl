using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime
using ThreadTools

function get_qnt(mdp::DenseGridWorld)  
  STATETYPE = statetype(mdp)
  ACTIONTYPE = actiontype(mdp)

  # This is an example state - it is a StaticArrays.SVector{2, Int}
  #s0 = SA[19,19]
      
  #       ####       Key            ####            Value     ####
  Q = Dict{Tuple{STATETYPE, ACTIONTYPE},            Float64}()
  N = Dict{Tuple{STATETYPE, ACTIONTYPE},            Int}()
  t = Dict{Tuple{STATETYPE, ACTIONTYPE, STATETYPE}, Int}()

  return Q, N, t
end

function init(seed=4)
  mdp = DenseGridWorld(seed=seed)
  Q, N, t = get_qnt(mdp)
  return mdp, Q, N, t
end 

############
# Question 2
############

function rollout(mdp::DenseGridWorld, policy_function, s0, max_steps=100)
  s = s0
  r_total = 0.0
  t = 0
  rewards = []
  while !isterminal(mdp, s) && t < max_steps
    a = policy_function(mdp, s) # policy
    #@show s, a
    s, r = @gen(:sp, :r)(mdp, s, a)
    push!(rewards, r)
    r_total += discount(mdp)^t * r
    t += 1
  end
  return r_total 
end

function euclidian_distance(x1, y1, x2, y2)
  return (sqrt((x2-x1)^2 + (y2-y1)^2))
end

function reward_distance_findmin(current_state, destinations)
  distances = zeros(length(destinations))
  for index in eachindex(destinations)
    distances[index] = euclidian_distance(
			  destinations[index][1], destinations[index][2], 
                          current_state[1],       current_state[2]
		       )
  end
  return findmin(distances)
end

function policy_heuristic(m,s)
  ### 1. identify closest goal
  ### 2. minimize the larger distance between x and y (cost be damned)
  
  reward_states = [ 
    [60,0], [60,20], [60,40], [60,60],
    [40,0], [40,20], [40,40], [40,60],
    [20,0], [20,20], [20,40], [20,60],
    [ 0,0], [ 0,20], [ 0,40], [ 0,60]
  ]

  mintuple = reward_distance_findmin(s, reward_states)
  #@show s, reward_states[mintuple[2]]

  x_dist = (reward_states[mintuple[2]][1] - s[1]) # positive if we are left of target
  y_dist = (reward_states[mintuple[2]][2] - s[2]) # positive if we are below target

  # Move to close the larger distance, approximates a straight line
  if abs(y_dist) > abs(x_dist) # then move in the y direction
    y_dist > 0  ? choice=:up    : choice=:down
  else # move in the x direction
    x_dist > 0  ? choice=:right : choice=:left
  end

  return choice
end

function calculate_sem(array)
  return (
    mean=mean(array), 
    sem=std(array) / sqrt(length(array)),
    entries=length(array)
  )
end

function policy_random(m, s)
  return rand(actions(m))
end

function run_problem_2()
  mdp = HW3.DenseGridWorld(seed=3)
  num_simulations = 500
  initial = rand(initialstate(mdp))

  results = [rollout(mdp, policy_random, initial) for _ in 1:num_simulations]
  @show calculate_sem(results)
  results = [rollout(mdp, policy_heuristic, initial) for _ in 1:num_simulations]
  @show calculate_sem(results)
end

############
# Question 3
############


### A few sentences describing problem3-tree after 7 iterations:
# 
# The tree at this stage is relatively shallow. The root node [19,19] should
# have a visit count of 6, and does. I would expect that few branches have 
# only been traversed once, but in this tree most of them are showing a count 
# of zero... 

function search(Q, N, S, A, state, exploration_bonus=1.0)
    if !(state in S)
      return nothing
    end
      
    for action in A
      if !haskey(N, (state,action))
        return action 
      end
    end

    state_actions = [(state, a) for a in A if haskey(N, (state, a))]

    scores = []
    for (s,a) in state_actions
      q = Q[(s,a)]
      
      exploration = 0
      visits = sum(N[(s,a)] for (s, a_i) in state_actions)
      exploration_term = log(visits) / N[(s,a)]
      if exploration_term > 0 
        exploration = exploration_bonus * sqrt(exploration_term)
      end
      
      push!(scores, q + exploration)
    end

    return state_actions[argmax(scores)][2]
end

function explore(Q, N, t, A, c)
  Ns = sum(N[(s,a)] for a in A)
  return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), A)
end

# r for recursion
function mctsr(mdp::HW3.DenseGridWorld, s, Q, N, t, d=10)
  S = states(mdp)
  A = actions(mdp)
  reward = 0

  if d ≤ 0
    return reward
  end

  #a = explore(Q, N, t, A, c)
  a = search(Q, N, S, A, s)
  if !haskey(N, (s, a))
    for a in A
      N[(s,a)] = 0
      Q[(s,a)] = 0
    end
    #return 
  end

  next_state, reward = @gen(:sp, :r)(mdp, s, a)
  q = reward + discount(mdp)*mctsr(mdp, next_state, Q, N, t, d-1)
  N[(s,a)] += 1

  #@show Q[(s,a)], N[(s,a)]
  Q[(s,a)] += (q-(Q[(s,a)]/N[(s,a)]))
  return q
end

function mcts(mdp::HW3.DenseGridWorld, start_state, Q, N, t, iterations=7)
  S = states(mdp)
  A = actions(mdp)

  # Run for fixed number of iterations
  for k in 1:iterations
    println("        MCTS iteration $k of $iterations...")
    state = start_state
    path = []

    while true
        action = search(Q, N, S, A, state)
        
        if !haskey(N, (state, action))
            break
        end

        # Sample next state and reward
        next_state, reward = @gen(:sp, :r)(mdp, state, action)
        
        # Update transition count
        if !haskey(t, (state, action, next_state))
          t[(state, action, next_state)] = 0
        end
        t[(state, action, next_state)] += 1
        
        # Add to path
        push!(path, (state, action, reward, next_state))
        
        # Move to next state
        state = next_state
    end
    
    # Expansion
    for action in A
      @assert(!haskey(N, (state,action)))
      N[(state, action)] = 0
      Q[(state, action)] = 0
    end
    
    # Simulation
    cumulative_reward = rollout(mdp, policy_heuristic, state)
    
    # Backpropagation
    for (state, action, reward, _) in reverse(path)
      N[(state, action)] = get(N, (state, action), 0) + 1
      cumulative_reward = reward + mdp.discount * cumulative_reward
      Q[(state, action)] = get(Q, (state, action), 0) + 
        (cumulative_reward - get(Q, (state, action), 0)) / N[(state, action)]
    end

  end
  return Q, N, t
end 

function simulate_with_mcts(mdp, Q, N, t, starting_state, num_steps, mcts_iterations)
  S = states(mdp)
  A = actions(mdp)
  state = starting_state
  total_reward = 0.0
  
  for step in 1:num_steps
    println("    step $step of $num_steps...")
    # Clear previous search results
    empty!(Q)
    empty!(N)
    empty!(t)
    
    # Run MCTS from current state to choose the next action
    mcts(mdp, state, Q, N, t, mcts_iterations)
    #mctsr(mdp, state, Q, N, t, mcts_iterations)
    
    # Get the best action based on visit counts
    best_action = nothing
    best_visits = -1
    
    for action in A
      if haskey(N, (state, action)) && N[(state, action)] > best_visits
        best_visits = N[(state, action)]
        best_action = action
      end
    end
    
    # If no action found, break out of the loop
    if best_action === nothing
      break
    end
    
    # Execute the best action
    next_state, reward = @gen(:sp, :r)(mdp, state, best_action)
    total_reward += reward
    
    # Check if terminal state reached
    if isterminal(mdp, next_state)
      break
    end
    
    # Move to next state
    state = next_state
  end
  
  #return total_reward, Q, N, t
  return total_reward
end

function run_problem_3()
  mdp, Q, N, t = init(4) # seed

  # This is an example state - it is a StaticArrays.SVector{2, Int}
  s0 = SA[19,19]
      
  reward = simulate_with_mcts(mdp, Q, N, t, s0, 1, 7) # run once

  inbrowser(visualize_tree(Q, N, t, s0), "firefox") 

  println("Tree information after 7 iterations:")
  for ((s, a), q_value) in Q
    if s == s0
      n_value = N[(s, a)]
      println("Action $a: Q = $(round(q_value, digits=4)), N = $n_value")
    end
  end
end

############
# Question 4
############

function run_problem_4()
  mdp, Q, N, t = init(4) # seed
  s0 = rand(initialstate(mdp))

  rewards = []

  num_steps = 100
  num_simulations = 100
  mcts_iterations = 1000
  
  mean_reward = 0.0
  std_error = 0.0
  
  for sim in 1:num_simulations
    println("Running simulation $sim of $num_simulations...")
    reward = simulate_with_mcts(mdp, Q, N, t, s0, num_steps, mcts_iterations)
    push!(rewards, reward)
  end
  
  # Calculate statistics
  @show rewards
  #mean_reward = mean(rewards)
  #std_error = std(rewards) / sqrt(length(rewards))
  mean_reward, std_error = calculate_sem(rewards)

  println("Results of MCTS Planner Evaluation:")
  println("Mean accumulated reward: $(round(mean_reward, digits=4))")
  println("Standard error of the mean: $(round(std_error, digits=4))")
end

# A starting point for the MCTS select_action function (a policy) 
# which can be used for Questions 4 and 5
function select_action(m, s)
  Q, N, t = get_qnt(m)
  q = 0

  start = time_ns()
  # alternatively, set 40ms time limit
  # while time_ns() < start + 40_000_000 
  #Threads.@threads for _ in 1:1000
  for _ in 1:1000
    q = mctsr(m, s, Q, N, t)
    #results = mcts(m, s, Q, N, t)
  end

  # select a good action based on q and/or n
  best_action = argmax(Q)
  #@show best_action

  return best_action[2] # the actual action
end

function run_problem_5()
  mdp = DenseGridWorld(seed=10)
  
  #@btime select_action(m, SA[35,35]) # aiming for 10-20ms
  select_action(mdp, SA[35,35]) # aiming for 10-20ms
  
  @show results = [rollout(mdp, select_action, rand(initialstate(mdp)), max_steps=100) for _ in 1:100]
end 

############
# Question 5
############

#HW3.evaluate(select_action, "lumi6265@colorado.edu")

# If you want to see roughly what's in the evaluate function (with the timing code removed), check sanitized_evaluate.jl

########
# Extras
########

# With a typical consumer operating system like Windows, OSX, or Linux, it is nearly impossible to ensure that your function *always* returns within 50ms. Do not worry if you get a few warnings about time exceeded.

# You may wish to call select_action once or twice before submitting it to evaluate to make sure that all parts of the function are precompiled.

# Instead of submitting a select_action function, you can alternatively submit a POMDPs.Solver object that will get 50ms of time to run solve(solver, m) to produce a POMDPs.Policy object that will be used for planning for each grid world.
