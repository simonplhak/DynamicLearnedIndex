# TODOs

- merit
  - total build time - trainng time
    - jak dlouho nam trva jen trenova, jak dlouho nam trva zbytek?
  - insert throughput	
- pustit na 100M
- degeneration
  - pustit 3x experiment (1x nechavam existujici modely, 1x pokazde zmenim model, 1x nase heuristika)
  - precist znovu paper, jesli se k tomuhle vyjadruji

---

- nprobe search strategy
  - DFS
- ? train index until it is not degenerated
- features
  - deletion of objects
- performance
  - ? insertion of objects in bulk
- search quality
  - ? NeuralLSH at the top levels, while LMI at the bottom ones?
    - https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md#rules-of-thumb
- statistics
  - from what levels are the found kNNs
  - memory usage, both decorator based + estimate from components of the framework
  - store stats about evaluated queries (what k and nprobe values were used, how many candidates were found from which level, ...)
  - vyuzit time measuremet dekorator k globalnimu trackingu casu pro vyhodnoceni ...
  - aggregate statistics about levels, buckets, and objects (occupation stats, ...)
  - SearchResult
    - extend about information at what level neighbors were found
    - utils for analyzing SearchResult

---

- ? use more sophisticated sampling technique
  - force the same number of sampled objects from each bucket?
- fixed-sized buckets - implement my own algorithm for situations where the clustering does not conform to maximal bucket sizes - essentially `train` method (is this framework or index specific?)
