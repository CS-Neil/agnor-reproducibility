# CCAgT v2 expanded-release audit

Status: complete; fixed-threshold evaluation gate passed for 12 nonoverlapping
patient/slide units (2026-09-21, Asia/Shanghai)

## Official provenance

- Dataset: `CCAgT: Images of Cervical Cells with AgNOR Stain Technique`
- DOI/version: `10.17632/wg4bpm33hj.2`
- Official page: <https://data.mendeley.com/datasets/wg4bpm33hj/2>
- Licence: CC BY-NC 3.0.
- Official aggregate archive: 3,548,726,325 bytes; SHA-256
  `eb9f01e8feae0029056dac2dfa6f5e9780692e1a54c0df1bbd33cd279b9eb557`.
- The reproducible downloader uses the official public API's 15 image archives
  and 15 mask archives and verifies each archive against its own published
  size and SHA-256. This avoids treating an incomplete aggregate ZIP as data.

The official README and COCO JSON have been downloaded and independently
verified against their API hashes:

- `raw/readme.md`: `2877db9ff071c9b3e1baefb93da4ef4e4039bb53f38b974bccc0b74cf65bb5de`
- `raw/CCAgT_COCO_OD.json`: `e72929019866f1d26b518094a10cf1049d23dc144478b754505f65169d9188b4`

## Independent-unit evidence

The landing page says 9,339 images from 15 slides and does not itself state a
one-slide-per-patient mapping. The official README is more explicit: it states
that the release contains images from 15 patients, and its filename section
says that the slide identifier differentiates both slides and patients. The
release's A--O identifiers can therefore be treated as 15 patient/slide units,
subject to the usual limitation that this remains a single-institution
expanded release rather than multi-centre external validation.

## Annotation and count checks

- COCO JSON contains 9,339 unique image records and 63,190 annotations.
- Slide counts from the COCO image records are A=1,311, B=561, C=385,
  D=2,125, E=506, F=318, G=249, H=650, I=309, J=261, K=1,503, L=396,
  M=254, N=248, O=263.
- The README table reports O=262 while its total and the COCO JSON imply
  O=263. This one-image documentation inconsistency must be retained in the
  audit rather than silently corrected.
- COCO categories 1--3 are Nucleus, Cluster, and Satellite, matching the three
  modeled biological labels in v1. Categories 4--7 add out-of-focus nuclei,
  overlapped nuclei, non-viable nuclei, and leukocyte nuclei. Any v2 semantic
  evaluation must prespecify how these additional labels are handled; they
  cannot be silently merged into the original target classes.
- Category instance counts are Nucleus=14,013, Cluster=25,710,
  Satellite=6,478, Nucleus out of focus=9,270, Overlapped nuclei=1,373,
  Non-viable nuclei=1,412, and Leukocyte nucleus=4,934. These sum to 63,190.
- No exact filename-stem overlap exists between the v1 metadata and the v2
  COCO records. Because v2 renamed slides and uses a different filename
  convention, this is not sufficient to establish image independence.

## Completed integrity and overlap gates

1. All 15 image archives and 15 mask archives match their official sizes and
   SHA-256 values. Archive topology reconciles exactly with the 9,339 COCO
   records: 9,339 images and 9,339 masks, all 1,600 x 1,200 pixels.
2. Every mask is single-channel mode `L`; observed values are exactly 0--7,
   matching the seven published categories plus background. Image/mask IDs
   pair one-to-one. The 6,478 category-3 COCO annotations yield 6,478 frozen
   satellite centers by the prespecified bounding-box-center rule.
3. Exact encoded-file, exact decoded-RGB, and likely JPEG-reencoded overlap
   counts are all zero. The broader dHash screen produced 1,672 candidates.
   Every candidate also had an identical field serial across the two naming
   schemes: 1,043 map v1 `slide_0165`/Patient A to v2 A, 546 map
   `slide_0166`/Patient B to v2 B, and 83 map `slide_0167`/Patient C to v2 C.
   Visual examples in `overlap_audit_examples.png` confirm identical field
   content under different color/contrast processing.
4. Patient/slide units A, B, and C are therefore excluded in full before any
   v2 model inference. Units D--O (12 patients/slides) have no match under any
   audit rule and exceed the prespecified minimum of three eligible units.

The fixed-threshold evaluation gate passes for D--O. These units support a
same-institution, nonoverlapping expanded-release analysis; they do not support
a claim of multi-centre external validation. Full machine-readable evidence is
stored in `audit_results.json`.
