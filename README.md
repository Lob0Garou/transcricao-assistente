# Transcrição Assistente

Ferramenta experimental em Python para **revisar transcrições ouvindo os trechos duvidosos**.
Whisper transcreve localmente; análise acústica, DeepSeek e JEV ajudam a comparar alternativas.
A decisão final é humana: nenhuma sugestão substitui palavras automaticamente.

```text
Áudio → Whisper (duas passadas) → divergências e alertas
                                  ↓
                   re-transcrições + DeepSeek (opcional)
                                  ↓
                   pontuação acústica + JEV (opcional)
                                  ↓
                   revisão: ouvir, escolher e exportar
```

## Recursos

- Tempos por palavra e detecção de idioma por canal.
- Alertas de possíveis alucinações, sobreposição e voz sem palavra reconhecida.
- Candidatos obtidos por re-transcrição e comparação acústica.
- Página local com reprodução por trecho, velocidade ajustável e isolamento de canais.
- Palpites de locutor pelo tom da voz, que você confirma ou corrige.
- Fila de um áudio por vez, cache e exportação de HTML, TXT e JSON.

DeepSeek revisa texto; JEV avalia contexto. Uma frase plausível pode estar errada no áudio.
Concordância entre modelos não comprova que uma palavra foi dita.

## Instalação

Requisitos: **Python 3.12**, `ffmpeg` e `ffprobe` disponíveis no `PATH`, espaço para os modelos
Whisper (baixados no primeiro uso). GPU NVIDIA é opcional; CPU usa INT8 e pode ser bastante lenta.

```powershell
git clone https://github.com/Lob0Garou/transcricao-assistente.git
cd transcricao-assistente
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe app.py
```

Para GPU, mantenha o driver NVIDIA atualizado e instale o extra CUDA:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[gpu]"
```

Com o ambiente ativado, `python app.py` ou `Abrir transcricao.bat` abre `http://127.0.0.1:8790`.
O servidor escuta somente na máquina local; não o exponha como serviço público.
Alternativa sem venv: `python -m pip install --target .deps ".[gpu]"` e `python -s app.py`.
Windows é o ambiente testado; outros sistemas não foram validados de ponta a ponta.

## Uso pela CLI

```powershell
python transcrever.py "caminho\audio.ogg" --sem-deepseek --sem-jev
python transcrever.py "caminho\audio.ogg" --idioma pt --sem-deepseek --sem-jev
python transcrever.py "caminho\audio.ogg" --modelo small --sem-candidatos --sem-deepseek --sem-jev
python transcrever.py "caminho\audio.ogg" --refazer
```

Veja `python transcrever.py --help`. O padrão detecta o idioma por canal; forçar um idioma diferente
da fala pode produzir texto incorreto. A pontuação dos candidatos usa `large-v3` mesmo quando o
modelo inicial é menor; `--sem-candidatos` evita essa etapa mais pesada.

## APIs opcionais e privacidade

O áudio é processado localmente. **DeepSeek recebe o texto completo da transcrição e suas dúvidas;
JEV recebe trechos de contexto e candidatos.** Use esses revisores apenas com permissão para enviar
o texto aos provedores. No app, habilite as opções correspondentes. Na CLI, os revisores são tentados
por padrão quando há chaves; `--sem-deepseek --sem-jev` impede essas chamadas.

As APIs podem cobrar por uso, inclusive retentativas e rotas alternativas. Não há teto monetário.
Configure arquivos de chave fora do repositório:

```powershell
$env:JEV_KEY_FILE = "C:\segredos\jev.txt"
$env:DEEPSEEK_KEY_FILE = "C:\segredos\deepseek.txt"
$env:DEEPSEEK_BASE = "https://seu-provedor.example/v1"
$env:DEEPSEEK_MODELO = "modelo-disponivel-na-sua-conta"
```

Cada arquivo deve conter sua chave. `.env` não é carregado automaticamente.
Sem `DEEPSEEK_BASE`, as rotas são OpenCode Zen, OpenRouter e NVIDIA, com os identificadores
definidos em `ta/deepseek.py`. Disponibilidade e acesso dependem do provedor; não foram revalidados
com chamadas reais nesta publicação. `DEEPSEEK_ROTAS` limita/reordena as rotas
(`opencode,openrouter,nvidia`); a chave avulsa é usada apenas na primeira rota selecionada.

O revisor também lê credenciais existentes em `~/.local/share/opencode/auth.json`, ou em
`OPENCODE_AUTH`. Outras variáveis: `DEEPSEEK_CHAVE_PROVEDOR`, `DEEPSEEK_ESFORCO`,
`DEEPSEEK_MAX_TOKENS`. JEV usa `https://api.typesafe.ai/v1/systemone` com `jev-latest`.

`entradas/` e `saidas/` guardam áudio, transcrição e relatórios **sem criptografia**, sem expiração
automática, e são ignorados pelo Git. O relatório inclui uma cópia reproduzível do áudio. Revise a
pasta antes de compartilhar: logs e resultados também podem conter texto e caminhos locais.

## Cache e saídas

O app identifica reenvios por SHA-256. Os candidatos têm caches separados conforme DeepSeek/JEV
estejam habilitados. Use `--refazer` ou “refazer do zero” após trocar modelos, configuração ou código.
Caches antigos de candidatos são ignorados. Na CLI, não reutilize a mesma pasta `--saida` para
áudios diferentes: o cache de ASR depende da pasta.

Em `saidas/<audio>/`: `revisao.html`, `transcricao.txt`, `dados.json`, áudio para reprodução,
`asr_*.json`, `cand_*.json`, `idioma.json` e, no app, `origem.json` e `locutores.json`.

## Verificação

Testes offline, sem chaves, áudios particulares ou download de modelos:

```powershell
python -m unittest discover -s testes -p "test_*.py" -v
python testes/teste_rotas_deepseek.py
```

Os scripts de avaliação acústica usam arquivos externos, não distribuídos:

```powershell
$env:TA_AUDIO_CURTO = "caminho\audio-curto.ogg"
$env:TA_AUDIO_LONGO = "caminho\audio-longo.ogg"
python testes/teste_pontuacao.py
$env:TA_AUDIO_TURNOS = "caminho\audio-turnos.ogg"
$env:TA_CACHE_TURNOS = "caminho\asr_large-v3_en.json"
python testes/teste_turnos.py
```

Esses scripts verificam casos específicos em inglês, com frases e tempos definidos no código;
arquivos arbitrários não substituem o conjunto original. São registros dos experimentos, não um
benchmark público reproduzível nem uma taxa geral de precisão.

## Limitações e contribuição

- PT-BR ainda não tem validação representativa; limiares precisam de calibração.
- Separação por tom é um palpite para duas vozes, não diarização geral.
- Sobreposição em mono, riso e tosse exigem escuta humana.
- Pontuação acústica não é probabilidade calibrada de acerto.
- Ainda não há benchmark público nem validação em outra máquina.

Abra uma issue com ambiente e passos de reprodução; nunca anexe chaves ou material privado.
Para contribuir, execute os testes offline e documente limitações. Um conjunto de áudio autorizado
com transcrição humana de referência é uma contribuição especialmente útil.

Licença [MIT](LICENSE).
