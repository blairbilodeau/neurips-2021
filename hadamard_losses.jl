using Hadamard


function hadamard_losses(good_experts)
    # loss generated as explained in the NormalHedge paper
    # based on the 2^d x 2^d hadamard matrix

    # hyperparams
    dim = 6           # hadamard matrix dimension
    k = good_experts  # number or good experts
    eps = 0.025       # good experts gap
    rounds = 32768    # time horizon

    # define
    A = float(hadamard(2^dim))[2:end, :]
    A = [-A; A]
    A = repeat(A, 1, div(rounds, 2^dim))
    A[1:k, :] = A[1:k, :] .- eps
    mn = minimum(A[:])
    mx = maximum(A[:])
    A = (A .- mn) ./ (mx - mn)
    A'
end


function hadamard_loss(loss_params)
    experts_loss = loss_params["losses"]
    t = loss_params["step"]
    rep_factor = loss_params["rep_factor"]
    repeat(experts_loss[t, :], rep_factor)
end
