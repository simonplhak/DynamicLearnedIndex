# TODOs

core functionality
- distribution shift detection and ?reporting?
  - degeneration
    - pustit 3x experiment (1x nechavam existujici modely, 1x pokazde zmenim model, 1x nase heuristika)
    - precist znovu paper, jesli se k tomuhle vyjadruji
- deletion of objects

additional things
- ? use more sophisticated sampling technique
  - use constant sample size to provide bounded model construction time?
  - force the same number of sampled objects from each bucket?

---

- add overview of all important stats from build and search
- pustit na 100M

---

- nprobe search strategy
  - DFS
- ? train index until it is not degenerated
- performance
  - ? insertion of objects in bulk
- search quality
  - ? NeuralLSH at the top levels, while LMI at the bottom ones?
    - https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md#rules-of-thumb
- statistics
  - memory usage -> estimate from components of the framework?
  - vyuzit time measuremet dekorator k globalnimu trackingu casu pro vyhodnoceni ...
  - aggregate statistics about levels, buckets, and objects (occupation stats, ...)

---

- fixed-sized buckets - implement my own algorithm for situations where the clustering does not conform to maximal bucket sizes - essentially `train` method (is this framework or index specific?)
