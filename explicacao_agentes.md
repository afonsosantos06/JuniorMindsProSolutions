# Processo de Deteção de Fraude: Do Orquestrador aos Subagentes

## 1. Visão Geral (O Fluxo de Processamento)
O `orchestrator.py` atua como o cérebro principal do sistema. Sempre que uma nova transação entra na "pipeline", o Orquestrador recebe um resumo em JSON compacto (com os atributos guardados em `_TOOL_FIELDS`) e utiliza um LLM (via LangChain Agent) para decidir os próximos passos e invocar autonomamente ferramentas. O LLM funciona com base no `_ORCHESTRATOR_SYSTEM_PROMPT`.

O processo desenrola-se da seguinte forma:

### Passo 1: Validação Determinística (`check_rules`)
O Orquestrador recebe uma instrução forte para chamar sempre a ferramenta `check_rules` primeiro. Esta ferramenta funciona num ficheiro Python local isolado de Inteligência Artificial para regras muito diretas e exatas, tais como:
- Levantamento de montantes superiores ao saldo.
- Transferências para a própria conta (auto-transferência).
- Discrepâncias no país do IBAN do cartão vs Residência identificada.
- Atividade a altas horas da noite (00:00 as 04:59).
- Transferências para destinatários inéditos em montantes 3 vezes superiores ao comum.

Resultado do Passo 1:
- Se o `check_rules` devolver fraude absoluta (`is_fraud=True`), o processo é parado e a ação é marcada logo como **FRAUD**.
- Se devolver avisos (`is_warning=True`), esse "sinal amarelo" passa para a fase seguinte.

### Passo 2: Triagem Tática de Especialistas (Subagentes)
As transações variam. Portanto, o orquestrador tenta invocar subagentes de acordo com a predefinição delineada. Esses Subagentes reagem como prompts próprios aos LLM:
- **Pagamentos Físicos (In-person):** Invoca o `check_geo` (avalia viabilidade de distância/velocidade) e o `check_behaviour` (para análise de perfil financeiro habitual).
- **E-Commerce:** Invoca o `check_profile`, `check_comms` (compara eventuais interações fraudulentas como phishing no histórico recente) e o `check_behaviour`.
- **Transferências não habituais ("Novos" recipientes):** Invoca `check_profile` e `check_behaviour`.
- **Débitos Diretos fora de horas estipuladas:** Invoca `check_behaviour` e `check_comms`.
- **Tipologia Regular (Salários/Rendas/Outros a horas normais):** O prompt dita que podem fiar-se que apenas os subagentes chamados ou a revisão do `check_rules` basta. 

### Passo 3: Decisão Final (Mecanismo de Votação)
Todos os subagentes, após lerem o seu contexto (pings geográficos, registos LLM), retornam um JSON final que contem `{"verdict": "...", "confidence": "0.X"}`. O Orquestrador tem depois ordens estritas para tentar agregar matematicamente e inferir fraude com os outputs de ferramentas:
- 2+ subagentes identificarem `FRAUD` (>70% certas) → **FRAUD**
- Apenas 1 subagente + o aviso prévio de `check_rules` de cima → **FRAUD**
- 1 subagente (muito convicto, >85% de certeza) à qual se preza sinal forte → **FRAUD**
- Caso nada disto se verifique nos blocos acionados → Falha transitória do sistema que passa a marcar com **LEGIT**.

---

## 2. Porque é que quase tudo é classificado como "Legit"? (Excesso de Falsos Negativos para Fraudes Reais)

Ao analisar a arquitetura dos agentes, regras descritas e os ficheiros subjacentes, foi possível identificar várias razões críticas para o teu algoritmo estar sempre na zona "segura" e debitar demasiados outputs amigáveis que contornam a Fraude ("Ache que seja tudo Legit"):

### A. Falhas Críticas de `UNCERTAIN` Nos Subagentes
Os Subagentes do teu sistema contêm lógica de salvaguarda que reverte as escolhas à posição "Não Tenho Certeza" (`UNCERTAIN`) se as bases completas não estiverem visíveis:
- **`check_geo`**: Se não tiver correspondência de base em GPS (o que carece facilmente porque o JSON de localizações é apenas pontual/algumas localizações na data), diz `UNCERTAIN`.
- **`check_profile`**: Se o _sender_ da transação não constar como um id cidadão completo (mas sim uma empresa/employer como muitos que vemos no teu CSV), logo preestabelece que "Não existindo perfil, eu não posso julgar" respondendo `UNCERTAIN`.
- **`check_comms` / `check_behaviour`**: Têm um mínimo de instâncias necessárias (ex: comportamentos se existirem menos de 3 envios para analisar, descarta a avaliação). 
Isto significa que mais de **dois especialistas raramente opinam ativamente com `FRAUD`** simultaneamente porque retornam passivamente como `UNCERTAIN`. Sendo assim, o passo de "Decisão Final" cai invariavelmente na condicional _"Otherwise -> LEGIT"_.

### B. O `check_rules` **não tem o decorador `@tool`** (Erro Estrutural)
No ficheiro `src/rules.py` (linha 30), a definição de função não ostenta o marcador `@tool` presente nas linhas dos teus outros Agentes em `src/agents/*.py`. No código associado (`src/agents/orchestrator.py`), compõe-se `ALL_TOOLS = [check_rules, ...]`. A incapacidade das bibliotecas do LangChain tratarem corretamente a injeção ao LLM de uma ferramenta sem o decorator obriga por norma ao Orquestrador (num "BlindSpot") a saltar/simular esse envio ou falhar silenciosamente. Mesmo com uma falha sem aborto de script puro (Exception block do Orchestrator), o fallback padrão que estipulaste devolve: `LEGIT`. O que torna as verificações diretas nulas.

### C. O Orquestrador é ingénuo quanto à sua Autonomia (Agents LLM Loop Escapism)
O Agent foi preenchido com um System Prompts extenuante de lógica booleana na qual exige que simule internamente comportamentos e deduza que tipo de passos ou scripts correr. Grande parte dos LLMs tendem a encurtar processos (LLM Laziness) quando instruídos para inferir etapas. Isto leva a que frequentemente "leiam os dados iniciais do utilizador, comparem com a história dos logs" e ditem a resposta logo num objeto sem abrirem qualquer sub-recurso.

### D. A Fraude de "Linha 41"
Na referência à "Linha 41", o item trata de uma transação específica tipo "Débito Direto" numa *Rideshare App* as 05h39. Essa foi uma das únicas em que as condições estritas (`"direct_debit" at unusual hour`) disparou de forma manual com que forças os subagentes ao trabalho. Lá perante uma hora irregular em relação à subatividade do utente, o LLM sentiu uma ligeira suspeita gerando um `FRAUD`. Quase toda a base do teu CSV tem depósitos regulares de salários transferidos e recebimento de habitação. Pelo prompt do orchestrador (ver Passo 2 na última linha), ordenaste que `"Para rotinas como as de Renda ou as de ordenado, que a check_rules serve..."` -> Acoplado ao insucesso do `check_rules` sem "tooling", o agente nunca encontrou razão formal e deduzia as dezenas de transações seguras de volta a um amigável `LEGIT`.
