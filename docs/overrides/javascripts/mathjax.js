window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document.addEventListener("DOMContentLoaded", function() {
  var mathElements = document.getElementsByClassName("arithmatex");
  for (var i = 0; i < mathElements.length; i++) {
    mathElements[i].style.display = "block";
  }
}); 