using Stan
using MCMCChains
using Statistics
using LinearAlgebra

function get_div(fit)
    sampler_params = get_sampler_params(fit, inc_warmup = false)
    return [x for y in sampler_params for x in y["divergent__"]]
end

function check_div(fit)
    divergent = get_div(fit)
    n = sum(divergent)
    N = length(divergent)
    println("$n of $N iterations ended with a divergence ($(100 * n / N)%)")
    if n > 0
        println("  Try running with larger adapt_delta to remove the divergences")
    end
end

function check_treedepth(fit, max_depth = 10)
    sampler_params = get_sampler_params(fit, inc_warmup = false)
    depths = [x for y in sampler_params for x in y["treedepth__"]]
    n = sum(x == max_depth for x in depths)
    N = length(depths)
    println("$n of $N iterations saturated the maximum tree depth of $max_depth ($(100 * n / N)%)")
    if n > 0
        println("  Run again with max_depth set to a larger value to avoid saturation")
    end
end

function check_energy(fit)
    sampler_params = get_sampler_params(fit, inc_warmup = false)
    no_warning = true
    for (chain_num, s) in enumerate(sampler_params)
        energies = s["energy__"]
        numer = sum((energies[i] - energies[i - 1])^2 for i in 2:length(energies)) / length(energies)
        denom = var(energies)
        if numer / denom < 0.2
            println("Chain $chain_num: E-BFMI = $(numer / denom)")
            no_warning = false
        end
    end
    if no_warning
        println("E-BFMI indicated no pathological behavior")
    else
        println("  E-BFMI below 0.2 indicates you may need to reparameterize your model")
    end
end

function check_n_eff(fit)
    fit_summary = describe(fit)
    n_effs = [x[4] for x in fit_summary["summary"]]
    names = fit_summary["summary_rownames"]
    n_iter = length(fit[:lp__])

    no_warning = true
    for (n_eff, name) in zip(n_effs, names)
        ratio = n_eff / n_iter
        if ratio < 0.001
            println("n_eff / iter for parameter $name is $ratio!")
            println("E-BFMI below 0.2 indicates you may need to reparameterize your model")
            no_warning = false
        end
    end
    if no_warning
        println("n_eff / iter looks reasonable for all parameters")
    else
        println("  n_eff / iter below 0.001 indicates that the effective sample size has likely been overestimated")
    end
end

function check_rhat(fit)
    fit_summary = describe(fit)
    rhats = [x[5] for x in fit_summary["summary"]]
    names = fit_summary["summary_rownames"]

    no_warning = true
    for (rhat, name) in zip(rhats, names)
        if rhat > 1.1 || isnan(rhat) || isinf(rhat)
            println("Rhat for parameter $name is $rhat!")
            no_warning = false
        end
    end
    if no_warning
        println("Rhat looks reasonable for all parameters")
    else
        println("  Rhat above 1.1 indicates that the chains very likely have not mixed")
    end
end

function check_all_diagnostics(fit)
    check_n_eff(fit)
    check_rhat(fit)
    check_div(fit)
    check_treedepth(fit)
    check_energy(fit)
end

function compile_model(filename, model_name = nothing)
    model_code = read(filename, String)
    if model_name === nothing
        model_name = basename(filename)
    end
    sm = nothing
    try
        sm = Stanmodel(name = model_name, model = model_code)
    catch
        sm = Stanmodel(name = model_name, model = model_code)
    end
    return sm
end
