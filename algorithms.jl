using SpecialFunctions
using Roots
using ProgressBars


function init(algo, tot_experts)
    algo_params = Dict()
    if algo == squint
        algo_params = squint_init(tot_experts)
    elseif algo == coinBetting
        algo_params = coinBetting_init(tot_experts)
    elseif algo == adaHedge
        algo_params = adaHedge_init(tot_experts)
        algo_params["weights"] = normalize(ones(tot_experts))
    elseif algo == hedge
        algo_params = hedge_init(tot_experts)
    elseif algo == care
        algo_params = care_init(tot_experts)
    elseif algo == carl
        algo_params = carl_init(tot_experts)
    elseif algo == normalHedge
        algo_params = normalHedge_init(tot_experts)
    elseif algo == abnormal
        algo_params = init_abnormal(tot_experts)
    elseif algo == abnormalCont
        algo_params = init_abnormalCont(tot_experts)
    end
    algo_params
end


function running(algo_update, loss_generating_fun, loss_params)
    rounds = loss_params["rounds"]
    tot_experts = loss_params["tot_experts"]

    # initialize variables
    algo_loss = zeros(rounds)
    weights = normalize(ones(tot_experts), 1)
    iter = ProgressBar(1:rounds)
    set_description(iter, string(algo_update))
    params = init(algo_update, tot_experts)

    # overwriting hyperparameters
    if haskey(loss_params, "c_hedge")
        params["eta"] = loss_params["c_hedge"]
    elseif haskey(loss_params, "c_ada")
        params["alpha^2"] = loss_params["c_ada"]
    end

    # run algorithm
    for t in iter
        loss_params["step"] = t
        losses = loss_generating_fun(loss_params)
        algo_loss[t] = dot(weights, losses)
        weights = algo_update(algo_loss[t], losses, params)
    end
    algo_loss
end


function squint_init(tot_experts)
    params = Dict()
    params["regret"] = zeros(tot_experts)
    params["variance"] = zeros(tot_experts)
    params
end


function squint(algo_loss, experts_loss, params)
    """Squint algorithm from 
        "Second-order Quantile Methods for Experts and Combinatorial Games",
        https://arxiv.org/abs/1502.08009
    """
    # update parameters
    inst_regret = algo_loss .- experts_loss
    params["regret"] += inst_regret
    params["variance"] += inst_regret .^ 2

    # retrieve parameters and update weights
    R = params["regret"]
    V = params["variance"]
    evidence =
        sqrt(pi) * exp.(R .^ 2 ./ (4 * V)) .*
        (erfc.(-R ./ (2 * sqrt.(V))) - erfc.((V - R) ./ (2 * sqrt.(V)))) ./ (2 * sqrt.(V))
    normalize(evidence, 1)
end


function coinBetting_init(tot_experts)
    params = Dict()
    params["theta"] = zeros(tot_experts)
    params["wealth"] = ones(tot_experts)
    params["w_bets"] = zeros(tot_experts)
    params["step"] = 1
    params
end


function coinBetting(algo_loss, experts_loss, params)
    """Coin-Betting algorithm based on KT estimator from 
        "Coin Betting and Parameter-Free Online Learning",
        https://arxiv.org/pdf/1602.04128.pdf
    """
    # retrieve parameters
    w_bets = params["w_bets"]
    theta = params["theta"]
    wealth = params["wealth"]
    step = params["step"]

    # compute reward
    reward = algo_loss .- experts_loss
    reward[w_bets.<=0] = max.(reward[w_bets.<=0], 0)

    # feed coin-betting base algorithms
    theta += reward
    wealth += reward .* w_bets
    w_bets = theta .* wealth / step
    raw_weights = max.(w_bets, 0)
    if sum(raw_weights) == 0
        tot_experts = length(raw_weights)
        weights = normalize(ones(tot_experts), 1)
    else
        weights = normalize(raw_weights, 1)
    end

    # update parameters
    params["step"] += 1
    params["w_bets"] = w_bets
    params["theta"] = theta
    params["wealth"] = wealth
    weights
end


function adaHedge_init(tot_experts)
    params = Dict()
    params["alpha^2"] = log(tot_experts)
    params["lambda"] = 0
    params["cum_losses"] = zeros(tot_experts)
    params
end


function adaHedge(algo_loss, experts_loss, params)
    """Adahedge algorithm from "Follow the leader if you can, Hedge if you must",
        https://arxiv.org/pdf/1301.0534.pdf
    """
    # update parameters
    params["cum_losses"] += experts_loss

    # retrieve parameters
    lambda_t = params["lambda"]
    cum_losses = params["cum_losses"]
    alpha_square = params["alpha^2"]
    weights = params["weights"]

    # update weights
    if lambda_t == 0
        delta_t = algo_loss - minimum(experts_loss)
    else
        delta_t = algo_loss + lambda_t * log(dot(weights, exp.(-experts_loss / lambda_t)))
    end
    lambda_t += delta_t / alpha_square
    weights = normalize(exp.(-(cum_losses .- minimum(cum_losses)) / lambda_t), 1)
    params["lambda"] = lambda_t
    params["weights"] = weights
    weights
end


function hedge_init(tot_experts)
    params = Dict()
    params["step"] = 0
    params["eta"] = sqrt(log(tot_experts)) * C_HEDGE
    params["cum_losses"] = zeros(tot_experts)
    params
end


function hedge(algo_loss, experts_loss, params)
    """Hedge algorithm with decreasing learning rate from 
        "On the optimality of the Hedge algorithm in the stochastic regime",
        https://jmlr.csail.mit.edu/papers/volume20/18-869/18-869.pdf 
    """
    # update parameters
    params["step"] += 1
    params["cum_losses"] += experts_loss

    # retrieve parameters
    step = params["step"]
    eta = params["eta"]
    cum_losses = params["cum_losses"]

    # update weights
    lr = eta / sqrt(step)
    normalize(exp.(-lr * (cum_losses .- minimum(cum_losses))), 1)
end


function care_init(tot_experts)
    params = Dict()
    params["c1"] = C1_CARE
    params["c2"] = C2_CARE
    params["cum_losses"] = zeros(tot_experts)
    params["step"] = 0
    params
end


function care(algo_loss, experts_loss, params)
    """FTRL-Care algorithm from "Relaxing the I.I.D. Assumption: Adaptively Minimax 
        Optimal Regret via Root-Entropic Regularization"
        https://arxiv.org/pdf/2007.06552.pdf
    """
    # update parameters
    params["cum_losses"] += experts_loss
    params["step"] += 1

    # retrieve parameters
    c1 = params["c1"]
    c2 = params["c2"]
    cum_losses = params["cum_losses"]
    step = params["step"]

    # define useful functions
    entropy_func = w -> -dot(w, log.(w))
    exp_weights = (x, lr) -> normalize(exp.(-lr * (x .- minimum(x))), 1)

    # update weights
    N = length(experts_loss)
    search_fun =
        lr -> (lr - c1 * sqrt((entropy_func(exp_weights(cum_losses, lr)) + c2) / step))
    lr_hat = find_zero(search_fun, (0, c1 * sqrt((log(N) + c2) / step) + 1e-8))

    # update params and return weights
    normalize(exp.(-lr_hat * (cum_losses .- minimum(cum_losses))), 1)
end


function carl_init(tot_experts)
    params = Dict()
    params["const"] = C_CARL
    params["cum_losses"] = zeros(tot_experts)
    params["step"] = 0
    params
end


function carl(algo_loss, experts_loss, params)
    """FTRL-Carl algorithm from subsection 4.1 in our paper.
    """
    # update parameters
    params["step"] += 1
    params["cum_losses"] += experts_loss

    # retrieve parameters
    c_ab = params["const"]
    cum_losses = params["cum_losses"]
    step = params["step"]

    # update weights
    N = length(experts_loss)
    search_fun =
        c -> log(
            sum(
                exp.(
                    .-(c_ab * (cum_losses .- minimum(cum_losses)) ./ sqrt(step) .+ c) .^ 2,
                ),
            ),
        )
    c_hat = find_zero(search_fun, (0, 2 * sqrt(log(N))))
    exp.(.-(c_ab * (cum_losses .- minimum(cum_losses)) ./ sqrt(step) .+ c_hat) .^ 2)
end


function normalHedge_init(tot_experts)
    params = Dict()
    params["cum_losses"] = zeros(tot_experts)
    params["algo_cum_loss"] = 0
    params
end


function normalHedge(algo_loss, experts_loss, params)
    """Normalhedge algorithm from "A parameter-free hedging algorithm".
        https://arxiv.org/pdf/0903.2851.pdf
    """
    # update parameters
    params["algo_cum_loss"] += algo_loss
    params["cum_losses"] += experts_loss

    # retrieve parameters
    algo_cum_loss = params["algo_cum_loss"]
    cum_losses = params["cum_losses"]
    tot_experts = length(cum_losses)
    regret = max.(algo_cum_loss .- cum_losses, 0)
    regret_square = regret .^ 2

    # update weights
    mxm = maximum(regret_square) / 2
    mnm = minimum(regret_square) / 2 + 1e-8
    search_fun = ct -> sum(exp.((regret_square) / (2 * ct))) - tot_experts * exp(1)
    c_hat = find_zero(search_fun, (mnm, mxm))
    weights = regret .* exp.(regret_square / c_hat) / c_hat
    normalize(weights, 1)
end


function init_abnormal(tot_experts)
    params = Dict()
    params["step"] = 0
    params["cum_losses"] = zeros(tot_experts)
    params["c2"] = 1 / sqrt(2)
    params
end


function abnormal(algo_loss, experts_loss, params)
    """Abnormal algorithm from example 3 in our paper.
    """
    # update parameters
    params["step"] += 1
    params["cum_losses"] += experts_loss

    # retrieve parameters
    cum_losses = params["cum_losses"]
    eta_t = sqrt(params["c2"] / params["step"])
    N = length(cum_losses)

    # update weights
    search_fun =
        lambda -> sum(exp.(max.(lambda .- eta_t * cum_losses, 0) .^ 2 / 2) .- 1) / N - 1
    mxm = maximum(cum_losses)
    mnm = minimum(cum_losses)
    lambda_t = find_zero(
        search_fun,
        (eta_t * mnm + sqrt(2 * log(2)), eta_t * mxm + sqrt(2 * log(2))),
    )
    (exp.(max.(lambda_t .- eta_t * cum_losses, 0) .^ 2 / 2) .- 1) / N
end
