# TODOs

- ? train index until it is not degenerated
- features
  - deletion of objects
- performance
  - ? insertion of objects in bulk
- search quality
  - ? NeuralLSH at the top levels, while LMI at the bottom ones?
- statistics
  - aggregate statistics about levels, buckets, and objects (occupation stats, ...)
  - SearchResult
    - extend about information at what level neighbors were found
    - utils for analyzing SearchResult

---

- ? use more sophisticated sampling technique
  - force the same number of sampled objects from each bucket?
- fixed-sized buckets - implement my own algorithm for situations where the clustering does not conform to maximal bucket sizes - essentially `train` method (is this framework or index specific?)
