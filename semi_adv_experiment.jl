using Plots
using LinearAlgebra
using Random
using LaTeXStrings
LinearAlgebra.BLAS.set_num_threads(6)
include("algorithms.jl")
include("semi_adv_losses.jl")
output_path = string(@__DIR__) * "/figures/"


function experiment(algorithms, rounds, loss_params)

    regrets = zeros(rounds, length(algorithms))
    tot_experts = loss_params["tot_experts"]

    for (ind, algo) in enumerate(algorithms)
        algo_loss = 0
        experts_cum_loss = zeros(tot_experts)

        # algorithm parameters
        weights = normalize(ones(tot_experts), 1)
        params = init(algo, tot_experts)

        # run algorithm
        iter = ProgressBar(1:rounds)
        set_description(iter, string(algo))
        for t in iter
            # generate losses
            loss_params["step"] = t
            losses = semi_adv_loss(loss_params)

            # incur loss and update algorithm
            current_loss = dot(weights, losses)
            weights = algo(current_loss, losses, params)

            # update results
            experts_cum_loss += losses
            best_expert = minimum(experts_cum_loss)
            algo_loss += current_loss
            regrets[t, ind] = algo_loss - best_expert
        end
    end
    regrets
end


function plot_regret(data, label, shape, lines, cols)
    rounds = size(data, 1)
    time_seq = [1; 1000:div(rounds, 10):rounds]

    for (ind, algo) in enumerate(algorithms)
        plot!(
            time_seq,
            data[time_seq, ind],
            label = label,
            linewidth = lw,
            linealpha = la,
            linestyle = lines[ind],
            markeralpha = ma,
            markersize = [1; ms * ((1000:div(rounds, 10):rounds) .% div(rounds, 10) .== 0)],
            markershape = shape,
            markerstrokecolor = cols[ind],
            color = cols[ind],
        )
    end
end


# algo hyperparams
C1_CARE = sqrt(2 / 3)
C2_CARE = 1
C_CARL = sqrt(2)
C_HEDGE = sqrt(8)

# experiment parameters
N = 1000 # number total experts                                             -- paper uses 1000
N2 = 500 # helper for half the size of the experts                          -- paper uses 500
N0 = 2 # number of good experts for semi-adversarial loss                   -- paper uses 2
delta0 = 0.6 # Delta_0 + 0.5 for N0=2 case                                  -- paper uses 0.6
delta = 0.1
T = 10000 # total time                                                      -- paper uses 10000
COLORS =
    [:black, :dodgerblue, :darkgoldenrod3, :darkorchid3, :red, :green, :orange, :yellow]
algorithms = [carl, hedge, care]


# helpers for plots
lw = 3
la = 0.5
la_adv = la * 1.5
ma = 0.5
ms = 6
ylim_vals = (-5, 150)
xlim_vals = (-200, 10100)
shapes = [:circle, :square, :utriangle]
lines = [:dot, :dash, :solid]
semi_cols = [:orangered, :dodgerblue3, :darkgreen]
adv_cols = [:red, :purple, :darkgreen]
stoch_cols = [:goldenrod, :deepskyblue, :green2]
colors = Dict("semi" => semi_cols, "adv" => adv_cols, "stoch" => stoch_cols)
ppp = plot(
    ylim = ylim_vals,
    xlim = xlim_vals,
    ylabel = "Expected Regret",
    xlabel = "Time",
    legend = :topleft,
)

# define loss function parameters
loss_fun = Dict()
loss_fun["tot_experts"] = N
loss_fun["eff_experts"] = N0
loss_fun["semi_gap"] = delta0
loss_fun["stoch_gap"] = delta
setting = ["semi", "adv", "stoch"]
for (ind, mod) in enumerate(setting)
    loss_fun["mod"] = mod
    res = experiment(algorithms, T, loss_fun)
    plot_regret(res, "hello", shapes[ind], lines, colors[mod])
end

output_path = "figures/"
savefig(ppp, output_path * "semi_adv_losses.png")
