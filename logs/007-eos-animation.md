# 007 — DRAFT — Edge of stability on the singular tanh surface, animated

- **Date:** 2026-07-31
- **Type:** DRAFT (diagram) — **exploratory**, no hypothesis. Alex asked to see what
  the oscillation looks like in the 3-D perspective. Recorded as exploratory per
  CONSTITUTION 1; the *central flows* question Alex raised alongside it is a separate
  experiment and is **not run** (see bottom).
- **Figure:** `figures/diagram/007_eos_animation.html`
- **Published:** https://claude.ai/code/artifact/26afee79-27f8-4e80-8bf4-72000a0ed11c
- **Seeds / settings:** data seed 1 (slider 1–40), minibatch RNG seed 20261.
  Defaults: $n=400$, $\sigma=0.3$, $\eta=0.05$, full batch, init $w = (1.2,\ 8.0)$.

## Setup

Same one-unit tanh net, but fitting a **finite noisy sample** instead of the
population. True function $f_0 = 0$; labels $y_i = \sigma\varepsilon_i$ are pure noise.
The empirical loss factorises exactly, the same way the population loss did:

$$L_n(w_1,w_2) = C - w_2 B(w_1) + \tfrac12 w_2^2 A(w_1)$$

with $A(a) = \overline{\tanh^2(a x_i)}$, $B(a) = \overline{y_i \tanh(a x_i)}$,
$C = \overline{y_i^2}/2$. Two 1-D functions of $w_1$ generate the whole 2-D landscape,
which is why the page can rebuild and animate live. $A, B$ and their first two
derivatives are analytic; checked against finite differences to $10^{-10}$ (gradient)
and $10^{-7}$ (sharpness).

$\sigma = 0$ leaves the cross intact ($B \equiv 0$). $\sigma > 0$ breaks it: the
minimum moves off $W_0$ to some $w \neq 0$.

## What happens

Textbook progressive sharpening into EoS, on a singular model, measured:

| $\eta$ | $2/\eta$ | final $w_2$ | final sharpness |
|---|---|---|---|
| 0.05 | 40 | 6.3046 | 39.747 |
| 0.02 | 100 | 9.9874 | 99.749 |

Sharpness climbs monotonically from below (it starts *negative* — the init is at a
saddle-ish point) and pins just under $2/\eta$, where it stays to 5 significant
figures.

The mechanism is legible on this model, which is the reason it is worth having:
sharpness at $(w_1, w_2)$ is $\tfrac12 w_2^2 g''(w_1)$, and $g''(0) = 2$, so on the
$w_1 = 0$ branch sharpness is exactly $w_2^2$. GD drives $w_1 \to 0$, which *raises*
sharpness — that is the progressive sharpening. Once $w_2^2 > 2/\eta$ the $w_1$
iteration has multiplier $|1 - \eta w_2^2| > 1$ and $w_1$ oscillates with growing
amplitude; the growing $|w_1|$ turns on the $w_2$ gradient
($\partial L/\partial w_2 = w_2 A(w_1)$), pushing $w_2$ down until $w_2^2 = 2/\eta$.
Self-stabilisation, with $w_2 \to \sqrt{2/\eta}$ as the fixed point. That is exactly
$6.3046 \approx \sqrt{40}$ and $9.9874 \approx \sqrt{100}$.

**Batch size changes where it settles**, and this is the part worth staring at
(measured at $\eta = 0.05$, $2/\eta = 40$, averaged over the last 1500 of 4000 steps):

| batch | $w_1$ oscillation amplitude | steady sharpness |
|---|---|---|
| full (400) | 0.0000 (decays out) | 39.77 |
| 64 | 0.135 | 33.76 |
| 16 | 0.207 | 26.12 |

Full batch reaches EoS and then the oscillation **decays** — the multiplier is
$|1-\eta\lambda|$ slightly below 1, so it is a converging 2-cycle. Persistent visible
oscillation needs gradient noise. With minibatches the oscillation never dies and
sharpness settles *below* $2/\eta$, further below as the batch shrinks. That gap is a
known effect (the stochastic sharpness gap), reproduced here on a two-parameter model.

## Rendering choices that change what you see

- **"Stretch axes to fit" is on by default and distorts the aspect ratio.** The $w_1$
  oscillation is order $0.1$ while the $w_2$ excursion is order $2$ — at true aspect
  the oscillation is a few pixels wide and invisible. The toggle turns the distortion
  off; the shape you see with it on is not the shape of the landscape.
- Default view is $w_1 \in \pm1.5$, $w_2 \in \pm11$ for the same reason.
- The trajectory is drawn **over** the surface rather than depth-sorted into it, so it
  stays visible when it passes behind the near edge. It is not occluded correctly.
- $z$ is linear in $L_n$, rescaled to the visible range of the current window.
- The sharpness trace shows every recorded step and rescales its $y$ axis to fit, so
  the $2/\eta$ line moves when $\eta$ changes. Changing $\eta$ mid-run does not restart.

## Open threads — the experiment that was NOT run

Alex asked "can we replicate central flows here?" That is an experiment under
CONSTITUTION 1 and needs a hypothesis on record first. Noting what it would involve so
the question is ready to answer:

Central flows (Cohen et al.) predict that at EoS the *time-averaged* GD trajectory
follows a modified gradient flow — gradient flow on $L$ plus a term that holds
$\lambda_{\max}$ pinned at $2/\eta$, with the oscillation amplitude determined
self-consistently rather than being noise. This model is unusually well suited to
testing it: two parameters, analytic derivatives, an exactly known EoS fixed point
$w_2 = \sqrt{2/\eta}$, and a top eigenvector that is essentially $\hat e_1$ throughout.
The comparison would be the averaged GD iterate against a numerically integrated
central flow, over a range of $\eta$.

Before it runs it needs, from Alex:

- **Hypothesis:** _(blank)_
- **What "replicated" would mean quantitatively** — agreement in the $w_2(t)$
  trajectory? in the predicted oscillation amplitude? to what tolerance?
- **The baseline** (CONSTITUTION 0a): plain gradient flow on $L$ already predicts the
  descent phase, and $w_2 \to \sqrt{2/\eta}$ is derivable in two lines from the
  self-stabilisation argument above without any central-flow machinery. Central flows
  has to beat *that*, not just fit the data.

Worth flagging: nothing in this entry is about SLT. It is a singular model, but every
number above came from the Hessian and none of it needed the RLCT. That is itself
relevant to the project's standing question.
