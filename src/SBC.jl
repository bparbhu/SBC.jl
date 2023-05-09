module SBC

using Stan
using DataFrames
using Random
using Statistics
using LinearAlgebra
using Distributed
using CSV

include("stan_utilities.jl")

const SEED = 1234

function timed(fn)
    function timed_wrapped(args...; kwargs...)
        start = time()
        r = fn(args...; kwargs...)
        finish = time()
        @info "$(fn) took $(finish - start)s"
        return r
    end
    return timed_wrapped
end

function model_name(filename)
    return basename(filename, ".stan")
end

function run_result(prior_data, stats, params)
    merge!(stats, Dict(string(k, "_prior") => prior_data[k] for k in keys(prior_data) if k in params))
    return stats
end

function compute_param_stat(data, fit_df, stat)
    params = intersect(keys(data), setdiff(names(fit_df), ["lp__"]))
    return Dict(k => stat(data[k], fit_df[:, k]) for k in params)
end

function order_stat(prior_val, posterior_samples)
    return sum(prior_val .< posterior_samples, dims=1)
end

function mean_stat(prior_val, posterior_samples)
    return mean(prior_val .< posterior_samples, dims=1)
end

function rmse_mean(prior, samples)
    return sqrt(sum((mean(samples) - prior) .^ 2))
end

function rmse_averaged(prior, samples)
    return sqrt(sum((samples - prior) .^ 2) / length(samples))
end

abstract type SBC end

function run_dgp(sbc, data, num_datasets)
    fit = stan_sample(sbc.dgp_model, data, iter=num_datasets,
                      warmup=0, chains=1, algorithm="Fixed_param",
                      seed=sbc.seed)
    return group_params(fit, data)
end

function fit_data(sbc, fit_data, sampler_args=nothing)
    if sampler_args !== nothing
        kwargs = merge(sbc.sampler_args, sampler_args)
    else
        kwargs = sbc.sampler_args
    end
    return stan_sample(sbc.fit_model, fit_data, seed=sbc.seed; kwargs...)
end

function check_fit(sbc, fit, summary)
    # Use stan_utility diagnostics here to check
end

function xform_fit(sbc, fit, summary)
    return fit, DataFrame(fit), summary
end

function compute_stats(sbc, data, fit_df)
    stat_dicts = [Dict(string(k, "_", nameof(stat)) => v
                      for (k, v) in compute_param_stat(data, fit_df, stat))
                  for stat in sbc.stats]
    return merge(stat_dicts...)
end

function get_summary_stats(summary, pars)
    result = Dict()
    for p in pars
        flatnames = [p2 for p2 in keys(summary) if startswith(p2, string(p, "["))]
        result[string(p, "_rhat")] = getindex.(Ref(summary), flatnames, "rhat")
        result[string(p, "_n_eff")] = getindex.(Ref(summary), flatnames, "n_eff")
    end
    return result
end

@timed function replication(sbc, fit_data)
    fit = fit_data(sbc, fit_data)
    summary = fit_summary(fit)
    check_fit(sbc, fit, summary)
    fit, df, summary = xform_fit(sbc, fit, summary)
    stats = compute_stats(sbc, data, df)
    params = intersect(keys(fit_data), setdiff(names(df), ["lp__"]))
    summary_stats = get_summary_stats(summary, params)
    merge!(stats, summary_stats)
    return run_result(fit_data, stats, params)
end

@timed function run(sbc, data, num_replications)
    gdata = run_dgp(sbc, data, num_replications)
    return DataFrame(collect(pmap(replication(sbc), gdata)))
end

mutable struct VanillaSBC <: SBC
    dgp_model_name::String
    fit_model_name::String
    sampler_args::Dict
    stats::Vector{Function}
    seed::Int
    dgp_model
    fit_model
end

function VanillaSBC(dgp_model_name, fit_model_name, sampler_args;
                    stats=[order_stat], seed=SEED)
    dgp_model = Stanmodel(name=model_name(dgp_model_name), model=dgp_model_name)
    fit_model = Stanmodel(name=model_name(fit_model_name), model=fit_model_name)
    return VanillaSBC(dgp_model_name, fit_model_name, sampler_args,
                      stats, seed, dgp_model, fit_model)
end

function vanilla_sbc_8schools(num_reps)
    school_data = Dict(:J => 8, :K => 2, :sigma => [15, 10, 16, 11,  9, 11, 10, 18])
    sbc = VanillaSBC("../code/gen_8schools.stan", "../code/8schools.stan",
                     Dict(:chains => 1, :iter => 2000, :control => Dict(:adapt_delta => 0.98)),
                     stats=[rmse_mean, rmse_averaged])
    stats = run(sbc, school_data, num_reps)
    CSV.write(string(sbc, ".csv"), stats)
    @info stats
end

if isdefined(Main, :IJulia) && Main.IJulia.inited
    num_reps = 10
else
    num_reps = parse(Int, ARGS[1])
end

vanilla_sbc_8schools(num_reps)

export 

end