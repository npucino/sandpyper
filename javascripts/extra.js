window.MathJax = {
    tex2jax: {
      inlineMath: [ ["$","$"], ["\\(","\\)"] ],
      displayMath: [ ["$$","$$"], ["\\[","\\]"] ]
    },
    tex: {
    inlineMath: [["$","$"],["\\(", "\\)"]],
    displayMath: [["$$","$$"],["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
