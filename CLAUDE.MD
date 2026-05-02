# Especificação do Sistema: PINN Engine (Physics-Informed Neural Networks)

## 1. Domínio do Problema
Sistema de simulação baseado em IA para resolver Equações Diferenciais (ex: Equação do Calor ou Navier-Stokes) utilizando Redes Neurais Informadas pela Física. O sistema deve dispensar o cálculo numérico tradicional pesando as Leis da Física diretamente na Função de Perda (Loss Function).

## 2. Stack Tecnológica
- **Linguagem**: Python 3.11+
- **Framework de IA**: PyTorch (facilita o cálculo de gradientes automáticos / autograd)
- **Visualização (EDA)**: Matplotlib / Seaborn
- **Ferramenta de Testes**: Pytest


## 3. Estrutura de Diretórios e Responsabilidades
- `src/model/`: Contém a arquitetura da Rede Neural (Multi-Layer Perceptron).
- `src/physics/`: Módulo crítico. Contém as equações diferenciais parametrizadas (a "física" do sistema).
- `src/loss/`: Cálculo de erro. Combina o erro dos dados (MSE) com o erro residual da física (PDE Loss).
- `src/data/`: Gerador de condições de contorno (Boundary Conditions - BC) e condições iniciais (Initial Conditions - IC).
- `tests/`: Bateria de testes unitários. **Nenhum módulo é criado sem um teste correspondente.**

## 4. Diretrizes de Engenharia (Anti-Vibe Coding)
- **TDD Rigoroso**: Os testes de autograd (cálculo de derivadas parciais) devem ser escritos antes da função de perda.
- **Determinismo**: Seeds aleatórios (PyTorch/Numpy) devem ser fixados nos testes para garantir reprodutibilidade.
