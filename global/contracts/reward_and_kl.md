# Reward, credit, and KL contract

Every optimizer update is formed from four Actor ranks. Each rank samples six
completions for six images (36 rows), giving 144 global trajectories.

The active scalar rewards are format, rating, reasoning, and soft-overlong.
Rating is a same-image group-relative margin reward. Reasoning is derived from
the deterministic E5 Judge delta after the Editor applies only the Actor's
`solution` field.

Two formal routing modes are released:

- `field_component_kl002`: format/rating/reasoning credit is restricted to its
  parsed field. Reasoning and rating each receive sampled-k3 component KL with
  beta 0.02. Global completion KL is disabled.
- `completion_global_kl002`: all four rewards credit every active non-padding
  completion token. Component KL is disabled. One sampled-k3 KL term with beta
  0.02 is added on the loss side over the active eligible completion. It is not
  inserted into the reward.

The runtime audit rejects a completion-global-KL step unless
`vf/global_completion_kl_apply_count == 1` and
`vf/component_kl_apply_count == 0`. It also rejects non-finite loss, gradient,
reward, or KL values and incomplete four-rank trajectory coverage.
