
function semi_losses(t, N, N0, Delta0)
    N02 = div(N0, 2)
    if iseven(t)
        losses_semi = [0 * ones(N02); 1 * ones(N02); Delta0 * ones(N - N0)]
    else
        losses_semi = [1 * ones(N02); 0 * ones(N02); Delta0 * ones(N - N0)]
    end
    losses_semi
end


function stoch_losses(N, Delta)
    [0.4; (0.4 + Delta) * ones(N - 1)]
end


function adv_losses(t, N)
    N2 = div(N, 2)
    if iseven(t)
        losses_adv = [0 * ones(N2); 1 * ones(N2)]
    else
        losses_adv = [1 * ones(N2); 0 * ones(N2)]
    end
    losses_adv
end


function semi_adv_loss(loss_params)
    mod = loss_params["mod"]
    if mod == "semi"
        t = loss_params["step"]
        N = loss_params["tot_experts"]
        N0 = loss_params["eff_experts"]
        Delta0 = loss_params["semi_gap"]
        return semi_losses(t, N, N0, Delta0)
    elseif mod == "adv"
        t = loss_params["step"]
        N = loss_params["tot_experts"]
        return adv_losses(t, N)
    else
        N = loss_params["tot_experts"]
        Delta = loss_params["stoch_gap"]
        return stoch_losses(N, Delta)
    end
end