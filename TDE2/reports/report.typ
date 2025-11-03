#set text(font: "Atkinson Hyperlegible")
#set par(
  justify: true,
  leading: 0.52em,
)
= TRABALHO CLASSIFICAÇÃO DE IMAGENS

Deep features (`vetor X_deep`) usando CNN (`Inception_v3` ou outra de sua escolha)

== Taxas de Acerto (%)
#align(center)[
#table(
  columns: (auto, auto),
    table.header(table.cell(colspan: 2, [*Taxa de Acerto*]   )),
    table.header([], [*Validação Cruzada (5 folds)*]),
    align: (x, y) =>
    if y == 0 { center } else { top },
    [ KNN (parâmetros)  ],[],
    [ Árvore Decisão (parâmetros)  ],[],
    [ SVM (parâmetros)], [],
    [ Naive Bayes], [],
    [ Random Forest], [],
    [ Bagging], [],
    [Xgboost], []
  )
]
- Apresentar a matriz de confusão do melhor modelo
- Apresentar exemplos de erros do melhor modelo

== Considerações Finais

- Discutir os pontos fortes e fracos do melhor modelo.