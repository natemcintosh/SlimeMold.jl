using Random
using Agents
using CairoMakie

@agent struct Slime(GridAgent{2})
    # A slime agent has a heading, which is a direction in radians
    heading_rad::Float64
end

function slime_step!(agent, model)
    # Based on the agent's current heading, pick a position in that direction
    # with some amount of randomness
    new_heading = agent.heading_rad + randn() * 0.1

    # Update the agent's heading
    agent.heading_rad = new_heading

    # Calculate the direction in which to walk. It must be a tuple of Ints
    direction = (round(Int, cos(new_heading)), round(Int, sin(new_heading)))

    # Have the agent walk in the new direction
    walk!(agent, direction, model)
end

# This is the step in which the particles deposit a pheromone in their current location,
# the pheromone diffuses, then decays.
function ground_step!(model)
    # === Pheromone deposition =================================================
    # For each neighbor
    for agent in allagents(model)
        # Deposit pheromone
        model.ground[agent.pos[1], agent.pos[2]] += 1
    end

    # === Pheromone diffusion ==================================================

    # === Pheromone decay ======================================================
end

function initialize(;
    total_agents::Int = 100,
    gridsize::Tuple = (1_000, 1_000),
    seed::Int = 42,
)
    space = GridSpace(gridsize; periodic = true)
    rng = Random.Xoshiro(seed)
    model = StandardABM(
        Slime,
        space;
        properties = (ground = zeros(Float64, gridsize),),
        rng = rng,
        container = Vector,
        agent_step! = slime_step!,
        model_step! = ground_step!,
    )

    # Populate the model with agents
    for _ in 1:total_agents
        add_agent!(model, rand(rng) * 2π)
    end

    model
end

function run_model!(model; n_steps::Int = 10)
    agent_data = [:pos]
    adf, mdf = run!(model, n_steps; adata = agent_data)
    adf
end

function animate(; seed = 42)
    model = initialize(seed = seed)
    abmvideo(
        "slime.mp4",
        model;
        frames = 10,
        title = "Slime mold simulation",
        agent_size = 5,
        agent_color = :green,
        # How do I make the background white?
        # agentsplotkwargs = (backgroundcolor = :white),
    )
end
