# Tracking Identity Severity Golden Cases

Date: 2026-07-28

These are hand-authored expectations. Production evaluator output was not used
to generate the expected values. Synthetic frame spacing is one second so
wrong-ID frame count equals wrong-ID matched seconds in these examples.

## Declared episode policy

The primary conservation unit is a connected pair-event: wrong-ID matched rows
within one video may be connected when they occur no more than 15 frames apart
and share a GT or predicted identity. A pair exchange can therefore be one
pair-event containing two GT-level identity-error episodes.

For the scientific interpretation in these fixtures:

- recovery requires a later authoritative correct match for every affected GT;
- recovery latency is the frame distance from the last wrong match to the first
  correct recovery match;
- terminal means wrong at the affected GT's final authoritative matched frame;
- permanent includes a terminal swap or a declared persistence threshold;
- unresolved-GT events are retained but excluded from authoritative rankings.

The production permanent/terminal taxonomy does not yet satisfy this policy.

## Worked cases

### A. Perfect continuity

`A A A A A` has no predicted-ID transition, wrong match, episode, permanent
swap, or terminal swap.

### B. One-frame wrong assignment

`A B A` creates two standard identity switch events: `A -> B` and `B -> A`.
Only the middle frame is wrong, so there is one one-second recovered episode.
It is neither permanent nor terminal.

### C. Longer wrong assignment with recovery

`A B B B A` also creates two IDSW events, but it has three wrong-ID frames and
a three-second episode. This proves that equal IDSW counts can have different
identity-error severity.

### D. Persistent terminal corruption

`A` followed by nine `B` matches creates only one IDSW event. It contributes
nine wrong-ID frames and remains wrong at the last authoritative matched frame,
so it is terminal and persistent/permanent. Its IDSW count is lower than case B
despite much greater severity.

### E. Two-animal exchange that remains

Both GT animals change predicted identities once. Standard IDSW is two and
there are ten wrong-ID matched rows over five wrong frames. Identity-connected
pair grouping yields one pair-swap event, while GT-primary grouping yields two
GT-level episodes. Summary fields must not accidentally report the one
pair-event as two permanent pair swaps.

### F. Two-animal exchange and switchback

Each GT changes to the other predicted identity and later changes back, giving
four standard IDSW events. Six wrong-ID rows form one recovered pair-event with
two GT-level episodes. It is not terminal.

### G. Loss, different re-entry ID, then recovery

`A miss miss B A` gives two IDSW events under the current gap-persistent policy:
`A -> B` on re-entry and `B -> A` on recovery. A strict reset-on-gap policy
would count only `B -> A`, giving one. The repository does not expose separate
strict and gap-tolerant IDSW metrics, so the second value is a declared
comparison contract, not a production output.

### H. Fragment without wrong identity

`A miss A` creates one strict fragment and no IDSW or wrong-ID duration.
Fragmentation is not itself an identity assignment error.

### I. Longer unmatched gap with the same ID

`A miss miss miss A` still has zero IDSW and zero wrong-ID duration. A miss may
affect FN and fragmentation but is not automatically an identity swap.

### J. Video boundary

One correct `A` sequence and one correct `B` sequence have independent
continuity and identity-assignment state. No switch or episode crosses the
video/session boundary.

### K. Hidden interval

The visible frames are correct `A` matches; the middle hidden frame is matched
as `B`. With `include_hidden=false`, the evaluated population has two correct
matches, zero IDSW, and zero wrong rows. With `include_hidden=true`, the
sequence is `A B A`, giving two IDSW events and one hidden wrong-ID row.

### L. Unresolved GT authority

The unresolved wrong-ID row remains in the audit population and is conserved
into an ambiguous event. It is excluded from authoritative permanent/terminal
ranking until GT authority is resolved.

## Severity conclusion

IDSW counts transition events and must be supplemented by duration and episode
diagnostics. The minimum interpretation distinguishes:

- identity switch event;
- temporary identity-error episode;
- persistent/permanent identity swap;
- terminal identity swap;
- wrong-ID matched duration.

The phrase `permanent IDSW` is not used because it is not a standard MOT metric.
