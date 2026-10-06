# Dataset V1 Plant-level split QC

## Assignments

- train: P02, P03, P05, P08, P09, P10, P11, P12, P13, P15, P18, P19, P20
- validation: P04, P07, P17
- test: P01, P14, P16

- Plant counts: {'train': 13, 'validation': 3, 'test': 3}
- Image counts: {'test': 24, 'train': 104, 'validation': 24}
- Point counts: {'test': 153, 'train': 658, 'validation': 152}

## Date distribution

```csv
split,acquisition_date,number_of_images,mean_node_count_per_image,number_of_node_points
test,2026-09-15,6,4.666666666666667,28
test,2026-09-18,6,5.333333333333333,32
test,2026-09-25,6,6.833333333333333,41
test,2026-09-29,6,8.666666666666666,52
train,2026-09-15,26,4.538461538461538,118
train,2026-09-18,26,5.423076923076923,141
train,2026-09-25,26,7.3076923076923075,190
train,2026-09-29,26,8.038461538461538,209
validation,2026-09-15,6,4.333333333333333,26
validation,2026-09-18,6,5.0,30
validation,2026-09-25,6,7.5,45
validation,2026-09-29,6,8.5,51
```

## Node count per image

```csv
split,mean,median,min,max
test,6.375,6.0,4,10
train,6.327,6.0,4,10
validation,6.333,6.0,4,10
```

## Visibility distribution

```csv
split,visibility,count,percent
test,partially_occluded,7,4.57516339869281
test,uncertain,4,2.6143790849673203
test,visible,142,92.81045751633987
train,partially_occluded,30,4.5592705167173255
train,uncertain,7,1.0638297872340425
train,visible,621,94.37689969604864
validation,partially_occluded,12,7.894736842105263
validation,visible,140,92.10526315789474
```

## Checks

- PASS: Plant leakage
- PASS: Filename leakage
- PASS: All images assigned
- PASS: All points assigned
- PASS: Expected image counts
- PASS: Expected point counts
- PASS: Date coverage

DATASET V1 PLANT-LEVEL SPLIT QC: PASS
