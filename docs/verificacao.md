# Verificação da publicação inicial — 08/10/2026

## Correções preparadas

- `.gitignore`: exclui configurações locais, ambientes, credenciais e artefatos de build.
- `Abrir transcricao.bat`: remove o caminho pessoal de chave.
- `app.py`: revisores externos desligados inicialmente, com indicação do envio de texto.
- `ta/env.py`: aceita venv e `.deps`; registra DLLs somente no Windows, mantendo seus handles.
- `ta/asr.py`: carregamento GPU/CPU compartilhado.
- `transcrever.py`: candidatos usam o fallback CPU; cache separado por revisores habilitados.
- `ta/juiz.py`: chave explicitamente configurada; falha total e respostas inválidas não contam como uso válido.
- `ta/deepseek.py`: valida envelope e estrutura JSON antes de aplicar sugestões.
- `pyproject.toml`: build instalável, dependências CUDA opcionais e licença declarada.
- `testes/teste_turnos.py` e `testes/teste_pontuacao.py`: arquivos externos configuráveis, sem caminhos pessoais;
  o teste acústico retorna código de erro quando falha.
- `testes/test_confiabilidade.py`: oito regressões offline com dados fictícios.
- `README.md`, `LICENSE`, `docs/post-linkedin.md`: documentação pública, MIT e texto para divulgação.

Os demais módulos da implementação original integram o primeiro commit sem alterações funcionais.

## Checks realizados

- Sintaxe dos 18 arquivos Python: aprovada.
- Oito testes offline de regressão: aprovados.
- Script de rotas DeepSeek, offline: aprovado.
- Quatro casos acústicos locais em inglês: aprovados.
- Avaliação local de turnos: cortes, separação de tom e concordância espectral aprovados.
- Geração de relatório com áudio local de 120 s e ASR em cache, sem candidatos e sem APIs: aprovada.
- HTTP local, página inicial, status e revisores desligados inicialmente: aprovados.
- `transcrever.py --help`: aprovado.
- Build de wheel com `pip wheel --no-deps`: aprovado.
- `git diff --cached --check`: aprovado.
- Revisão da lista publicada e busca por padrões de segredos: sem valores reais detectados;
  as correspondências nos testes de rotas são fixtures fictícias.

## Limites da evidência

Ambiente: Windows, Python 3.12.10, dependências locais já instaladas. O build do wheel não valida
uma instalação completa em máquina limpa. Fallback CPU testado com mock, não com uma transcrição
completa em CPU. Não houve nova transcrição ASR completa nem chamadas reais de API nesta rodada.

Áudios, caches e relatórios usados nos checks não são distribuídos. Os checks acústicos locais não
formam um benchmark público reproduzível e não medem precisão geral. PT-BR e outros sistemas
operacionais permanecem sem validação representativa. Não há pipeline de CI nesta publicação inicial.
