using Random
using Test
using Agents
using CairoMakie
using ImageFiltering

@agent struct Slime(GridAgent{2})
    # A slime agent has a heading, which is a direction in radians
    heading_rad::Float64
end

const polygon = Makie.Polygon(Point2f[(-1,-1),(2,0),(-1,1)])
function slime_marker(s::Slime)
    rotate_polygon(polygon, s.heading_rad)
end
"""
    wrap_idx(idx, max_idx)

If an index is out of bounds, wrap it around to the other side of the grid by
the right amount. Uses 1-based indexing.
"""
@inline function wrap_idx(idx, max_idx)
    if idx < 1
        return idx + max_idx
    elseif idx > max_idx
        return idx - max_idx
    else
        return idx
    end
end

@testset "wrap_idx" begin
    @test wrap_idx(0, 10) == 10
    @test wrap_idx(11, 10) == 1
    @test wrap_idx(5, 10) == 5

    @test wrap_idx(-1, 3) == 2
    @test wrap_idx(-2, 3) == 1
end

function slime_step!(agent, model)
    # Sense the pheromone concentration in the three squares in front of the agent,
    # and pick the one with the highest concentration.
    # The three in "front" of the agent are the in the direction of the agent's heading
    # and the two squares to the left and right of that direction.
    front = (
        wrap_idx(agent.pos[1] + round(Int, cos(agent.heading_rad)), size(model.ground, 1)),
        wrap_idx(agent.pos[2] + round(Int, sin(agent.heading_rad)), size(model.ground, 2)),
    )
    left = (
        wrap_idx(
            agent.pos[1] + round(Int, cos(agent.heading_rad + π / 2)),
            size(model.ground, 1),
        ),
        wrap_idx(
            agent.pos[2] + round(Int, sin(agent.heading_rad + π / 2)),
            size(model.ground, 2),
        ),
    )
    right = (
        wrap_idx(
            agent.pos[1] + round(Int, cos(agent.heading_rad - π / 2)),
            size(model.ground, 1),
        ),
        wrap_idx(
            agent.pos[2] + round(Int, sin(agent.heading_rad - π / 2)),
            size(model.ground, 2),
        ),
    )
    # Get the pheromone concentration in each of the three squares
    front_concentration = model.ground[front[1], front[2]]
    left_concentration = model.ground[left[1], left[2]]
    right_concentration = model.ground[right[1], right[2]]

    # Pick the square with the highest concentration
    if front_concentration >= left_concentration &&
       front_concentration >= right_concentration
        # Do nothing, the agent will continue in the direction it was heading
    elseif left_concentration >= front_concentration &&
           left_concentration >= right_concentration
        # Turn left
        agent.heading_rad += π / 2
    else
        # Turn right
        agent.heading_rad -= π / 2
    end

    # Calculate the direction in which to walk. It must be a tuple of Ints
    direction = (
        round(Int, cos(agent.heading_rad)) * model.particle_speed,
        round(Int, sin(agent.heading_rad)) * model.particle_speed,
    )

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
        model.ground[agent.pos[1], agent.pos[2]] += max(1, model.particle_deposit_amount)
    end

    # === Pheromone diffusion ==================================================
    imfilter!(model.buffer, model.ground, Kernel.gaussian(model.diffusion_rate))
    copy!(model.ground, model.buffer)

    # === Pheromone decay ======================================================
    model.ground .*= model.decay_rate
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
        properties = (
            ground = zeros(Float64, gridsize),
            buffer = zeros(Float64, gridsize),
            diffusion_rate = 1,
            decay_rate = 0.80,
            particle_speed = 2,
            particle_deposit_amount = 1,
            wander_rate = 0.1,
        ),
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
    model_data = [:ground]
    adf, mdf = run!(model, n_steps; adata = agent_data, mdata = model_data)
    adf, mdf
end

function animate(; seed = 42)
    model = initialize(total_agents = 2, gridsize = (200, 200), seed = seed)

    heatkwargs = (
        colormap = :grays,
        # colorrange = (0, 100),
        # colorbar = false,
    )

    abmvideo(
        "slime.mp4",
        model;
        agent_marker = slime_marker,
        frames = 100,
        framerate = 10,
        title = "Slime mold simulation",
        agent_color = :white,
        heatarray = :ground,
        heatkwargs,
        # How do I make the background white?
        # agentsplotkwargs = (backgroundcolor = :white),
    )
end
