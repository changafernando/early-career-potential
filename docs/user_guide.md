# User guide for the front office, scouts and coaches

## What a score means

A score of **40%** means that, of past players whose first seasons looked like this player's, about 4 in 10
went on to become high-impact contributors in career years 4–8. It is **not** a verdict on the player, and it
cannot see injuries, role, fit, character or work ethic. Those are the scout's and coach's job.

The **80% range** (for example 30–52%) shows how much the score moves if the model is retrained on slightly
different history. Wide ranges mean the evidence is thin, so lean more on scouting.

**Insufficient evidence** means the player has under 500 NBA minutes. The model does not score him, because
late bloomers often look like this (Christian Wood had 238 minutes after three seasons and became a contributor).

**Models disagree** means the main model and a different-family check model differ by 15 points or more. The
analyst reviews the case before the brief is used.

## The threshold

A player is **flagged for review** when his score is at least the threshold (0.25 today). The threshold comes from
the steering committee's judgement that missing a future contributor costs about three times as much as a
wasted offer: 1 ÷ (1 + 3) = 0.25. In the 2015-18 replay this caught 41 of 46 contributors, with 29 false flags.
Use the **Threshold explorer** tab to see what other cost ratios would do.

A flag means *look closely*, not *extend*. A player below the threshold can still be extended. Record why.

## The business case tab

It prices the model's mistakes in the extension decision. A missed contributor costs the amount on the slider;
a false flag costs that amount divided by the cost ratio. The counts of mistakes come from the 2015-18 replay.
The prices are placeholders until finance supplies real ones, and the comparison is with draft order, not
with your scouts. Read the payback as the most the model could add, not a promise.

## Using a brief in a meeting

1. **Score and status.** Where does he sit against the threshold and within his class?
2. **What moves the score.** Do the drivers match what scouts see? If the model rewards something scouts
   distrust (for example, a high free-throw rate built on a small sample), say so.
3. **Comparables.** Five past players with the most similar first seasons, and what happened to them.
4. **Scout and coach view.** Add the context the model cannot see. This field is required before any decision.
5. **Decision and log.** Record the decision, who made it and whether it agrees with the model. **An override
   needs a reason.** Overrides are reviewed against outcomes each season, so the model and the scouting
   process both learn.

## Rules

* Screening input only. The GM decides extension and trade calls.
* Scores are confidential to the front office and expire each season. Never share them with players, agents,
  the media or other teams.
* If you think a score is unfair to a player or group, raise it with Player Relations. The Model Review Board
  can suspend use.

## Who to ask

| Question | Contact |
|---|---|
| How to read a brief, a driver or a comparable | Franchise analyst |
| Changing the outcome definition or the cost ratio | Steering committee (via the Assistant GM) |
| Fairness, privacy or player-equity concerns | Player Relations / Model Review Board |
