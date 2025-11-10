#set text(font: "Atkinson Hyperlegible")
#set text(lang: "pt")
#set text(region: "br")
#set par(
  justify: true,
  leading: 0.52em,
)
= TRABALHO CLASSIFICAÇÃO DE IMAGENS
Iniciamos com a base de dados crua, extraímos features utilizando `Inception_v3` e separamos em 80/20. Utilizamos 20% para encontrar os melhores hiperparâmetros e então treinamos os modelos com os restantes 80%. Abaixo segue a taxa de acerto média dentre os folds para cada modelo.

#align(center)[== Taxas de Acerto (%)]
#align(center)[
#table(
  columns: (auto, auto),
    table.header(table.cell(colspan: 2, [*Taxa de Acerto Média*]   )),
    table.header([], [*Validação Cruzada (5 folds)*]),
    align: (x, y) =>
    if y == 0 { center } else { top },
    [ KNN (parâmetros)  ],[92%],
    [ Árvore Decisão (parâmetros)  ],[67%],
    [ SVM (parâmetros)], [97%],
    [ Naive Bayes], [92%],
    [ MLP], [96%],
    [ Random Forest], [96%],
    [ Bagging], [95%],
    [Xgboost], [95%],
    [AdaBoost], [80%]
  )
]
#linebreak()

== Melhor modelo
O modelo que obteve a melhor performance foi o SVM com uma acurácia de 97% na validação cruzada com 5 folds.
#figure(image("../figures/confusion_matrix_best_model.png", width: 70%),
  caption: [Matrix de confusão da SVM.],
)
=== Exemplos de erros do modelo.
Analisando alguns dos erros dos modelos, podemos observar algumas coisas interessantes:

#grid(
  columns: 3,
  gutter: 1em,  // Optional spacing between images
  [#align(horizon)[#figure(image("../data/raw/Base/humanos/73.jpg", width: 100%), caption: [Real: Humano #linebreak() Previsto: Praia])<erro1>]], 
  [#align(horizon)[#figure(image("../data/raw/Base/comida/966.jpg", width: 100%), caption: [Real: Comida #linebreak() Previsto: Flores])<erro2>]],
  [#align(horizon)[#figure(image("../data/raw/Base/elefante/555.jpg", width: 100%), caption: [Real: Elefante #linebreak() Previsto: Praia])<erro3>]],
)

No caso da @erro1, o modelo identificou o que aparenta ser uma imagem com a classificação incorreta. Já na @erro2, podemos observar que certas características da imagem remetem às flores (cores vibrantes, fundo com folhas verdes) o que pode ter gerado um embedding similar ao de flores. Finalmente, na @erro3 observamos um erro mais difícil de ser racionalizado, é possível que a areia e água presentes na imagem levaram à esta classificacão, no entanto é impossível dizer. 

== Considerações Finais

Como a `Inception_v3` fundamentalmente transforma as imagens em vetores o problema se torna separar esses vetores em suas respectivas classes, algo que a SVM lida bem por ser capaz de dividir esse espaço de alta dimensão. No entanto, erros ocorrem em casos como na @erro2 onde o vetor tem uma maior ambiguidade. Portanto, a SVM lida bem com problemas com alta distinção  e menor ambiguidade entre as classes. 