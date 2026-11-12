"""Referencias bibliográficas compartidas por los notebooks (formato APA, con DOI cuando existe).

Cada documento cita con la forma corta (Autor, año) y termina con `bloque([...])`, la lista de las
referencias que usa. Los datos se verificaron contra el registro del DOI (Crossref) o la página
de la revista.
"""

REFS = {
    "qiu2006": "Qiu, H., Lee, J., Lin, J. y Yu, G. (2006). Wavelet filter-based weak signature detection method and its "
               "application on rolling element bearing prognostics. *Journal of Sound and Vibration, 289*(4–5), 1066–1090. "
               "https://doi.org/10.1016/j.jsv.2005.03.007",
    "nasa_ims": "Lee, J., Qiu, H., Yu, G., Lin, J. y Rexnord Technical Services (2007). *IMS Bearing Data* [conjunto de "
                "datos]. NASA Prognostics Data Repository, NASA Ames Research Center.",
    "nelson1984": "Nelson, L. S. (1984). The Shewhart control chart—Tests for special causes. *Journal of Quality "
                  "Technology, 16*(4), 237–239. https://doi.org/10.1080/00224065.1984.11978921",
    "we1956": "Western Electric Company (1956). *Statistical Quality Control Handbook*. Western Electric Co.",
    "montgomery": "Montgomery, D. C. *Introduction to Statistical Quality Control* (8.ª ed.). Wiley.",
    "randall2011": "Randall, R. B. y Antoni, J. (2011). Rolling element bearing diagnostics—A tutorial. *Mechanical Systems "
                   "and Signal Processing, 25*(2), 485–520. https://doi.org/10.1016/j.ymssp.2010.07.017",
    "youden1950": "Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer, 3*(1), 32–35. "
                  "https://doi.org/10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3",
    "holm1979": "Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of "
                "Statistics, 6*(2), 65–70. https://www.jstor.org/stable/4615733",
    "bh1995": "Benjamini, Y. y Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful "
              "approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300. "
              "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x",
    "efron1993": "Efron, B. y Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC. "
                 "https://doi.org/10.1007/978-1-4899-4541-9",
    "wang2020": "Wang, B., Lei, Y., Li, N. y Li, N. (2020). A hybrid prognostics approach for estimating remaining useful "
                "life of rolling element bearings. *IEEE Transactions on Reliability, 69*(1), 401–412. "
                "https://doi.org/10.1109/TR.2018.2882682",
    "nectoux2012": "Nectoux, P., Gouriveau, R., Medjaher, K., Ramasso, E., Chebel-Morello, B., Zerhouni, N. y Varnier, C. "
                   "(2012). PRONOSTIA: An experimental platform for bearings accelerated degradation tests. En *IEEE "
                   "International Conference on Prognostics and Health Management (PHM'12)*, Denver. "
                   "https://hal.science/hal-00719503",
    "helwig2015": "Helwig, N., Pignanelli, E. y Schütze, A. (2015). Condition monitoring of a complex hydraulic system "
                  "using multivariate statistics. En *2015 IEEE International Instrumentation and Measurement Technology "
                  "Conference (I2MTC)* (pp. 210–215). https://doi.org/10.1109/I2MTC.2015.7151267",
    "breiman2001": "Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. "
                   "https://doi.org/10.1023/A:1010933404324",
    "liu2008": "Liu, F. T., Ting, K. M. y Zhou, Z.-H. (2008). Isolation forest. En *2008 Eighth IEEE International "
               "Conference on Data Mining* (pp. 413–422). https://doi.org/10.1109/ICDM.2008.17",
    "pedregosa2011": "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. "
                     "(2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*(85), "
                     "2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html",
}


def bloque(claves):
    """Sección de referencias en Markdown, en orden alfabético."""
    lineas = sorted(REFS[c] for c in claves)
    return "## Referencias\n\n" + "\n\n".join(lineas)
