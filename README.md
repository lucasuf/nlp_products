## ANÁLISE DE DADOS TEXTUAIS DE PRODUTOS PARA CONSTRUÇÃO DE CLASSIFICADOR DE IRREGULARIDADESANÁLISE DE DADOS TEXTUAIS DE PRODUTOS PARA CLASSIFICAÇÃO DE IRREGULARIDADES
O objetivo deste projeto é construir um modelo de classificação Multinomial Naive Bayes para evitar o cadastro de produtos irregulares

### Arquivos/pastas: 
- `/data`: dados brutos, ou exemplos de dados, para utilização dos códigos construidos.
- `/labeled_data`: dados rotulados em `illegal (0 or 1)`
- `/mercadolivre`: rotinas utilizadas no web scraping de produtos
- `/reports`: contém as métricas do modelo e análises de frequência das palavras
- `/build_model.py`: cria o modelo Multinomial Naive Bayes
- `/build_labeled_data.py`: processamento/tratamento dos dados
- `/run.py`: processa novas entradas com o modelo construído e gera reports
- `/utils.py`: métodos utilizados nas demais implementações

### Instruções
Para instalação dos requisitos:
```
pip install - requirements.txt
```
Para gerar os arquivos rotulados, caso os dados de origem mudem:
```
python build_labeled_data.py
```
Na construção do modelo:
```
python build_model.py
```
