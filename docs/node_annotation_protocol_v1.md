# Node annotation protocol v1

## Biological target

Annotate only nodes on the target pea plant's single main vertical stem. This cultivar does not branch laterally for the purposes of this study. A node is a point/keypoint at the visually identifiable main-stem junction where its corresponding leaf structure originates.

## Inclusion and numbering

- Include visible leaves, petioles, stems, and tendrils only insofar as they help locate the main-stem node junction.
- Exclude all structures from neighbouring plants, background vegetation, trellis/netting, greenhouse equipment, substrate, and unrelated objects.
- Number nodes in biological order from base toward apex: `Node 1`, `Node 2`, …, `Node N`.
- Do not annotate the undeveloped apical growing point. Annotate the apex only once a distinct developed node/junction is visible.

## Visibility and occlusion

For every point record one visibility state: `visible`, `partially_occluded`, or `uncertain`.

- `visible`: the junction is directly identifiable.
- `partially_occluded`: the junction location is still defensible despite overlap or obstruction.
- `uncertain`: the junction cannot be localized with confidence.

Never invent coordinates for a completely invisible node. Use the annotation tool's appropriate omission/visibility mechanism instead.

## Multi-view consistency

The same biological node may occur in multiple views of the same plant. Retain the same node index whenever it can be reliably matched. Do not force a match when visibility or identity is uncertain; record the ambiguity for review.

## P06 special case

P06 is biologically dead. Retain it in metadata with `plant_status = dead`; its intentionally low image count is not automatically missing/corrupt acquisition data.

## Annotation QA

1. Confirm target plant identity before placing points.
2. Trace the main stem base-to-apex and verify monotonically ordered node indices.
3. Review every partially occluded or uncertain point at high zoom.
4. Compare complementary views within the same plant/date for index consistency.
5. Have a second annotator or supervisor review an initial subset and resolve disagreements in a change log.

## Versioning

Protocol version: `v1`. Store the protocol version, annotator, annotation date, tool version, and any decision-rule change with each annotation release. Increment the version before changing biological definitions, visibility rules, or indexing conventions.
