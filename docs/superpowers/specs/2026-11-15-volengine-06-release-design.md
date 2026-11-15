# VolEngine — plano 6: empacotamento, wrapper C++ e 1.0 (design)

Data: 2026-11-15. Continua `docs/superpowers/specs/2026-11-10-volengine-design.md`: §5 (regras de estabilidade da
interface C), §7 (Python), §11 (plataformas, distribuição) e §13 (pastas). Tudo o que a spec-mãe fixa vale aqui sem
repetir. Os planos 1 a 5 estão fechados, e o autor considera as pendências deles resolvidas.

## 1. Intenção e o porquê

A 1.0 congela a interface C. Depois dela, nenhuma assinatura muda e nenhum valor de enum some (§5). O primeiro público
da 1.0 é quem faz `pip install volengine` e usa em Python. O wrapper C++ e os binários avulsos da release vêm depois
em prioridade, mas o wrapper entra na 1.0 (decisão do autor).

Sucesso é:
- `pip install volengine` funciona nas 7 plataformas da §11 sem compilador;
- o wheel instalado tem a mesma velocidade do build local no mesmo hardware;
- um projeto C++ consome o wrapper por `FetchContent` ou `find_package`;
- toda mudança na ABI depois da 1.0 é pega pelo CI.

## 2. Objetivos, não objetivos, antiobjetivos

**Objetivos**
- Wheels `py3-none-<plataforma>` para Windows x64, macOS arm64 e x64, Linux glibc e musl em x64 e arm64, no PyPI.
- Biblioteca nativa de cada plataforma anexada à GitHub Release, com SBOM.
- Python no uv (desenvolvimento, CI e release).
- Wrapper C++23 header-only `volengine.hpp` sobre a interface C, com pacote CMake.
- Contrato da ABI versionado e conferido no CI; abidiff no Linux.
- Release `v0.9.0` de ensaio antes da `v1.0.0`.

**Não objetivos**
- sdist. Sem wheel para a plataforma, o pip informa que não há distribuição em vez de tentar compilar (decisão do
  Claude; a §11 lista só wheels).
- Windows arm64, Python abaixo de 3.11 e outros wrappers (C#, R, Rust).
- Site de documentação. A 1.0 leva README e guia rápido (§6).
- Funções novas na interface C, salvo o que a auditoria da ABI (§6) ou o wrapper (§5) mostrarem que falta.

**Antiobjetivos**
- Wheel mais lento que o build local no mesmo hardware (por exemplo, por runtime ou flags errados).
- Wheel que instala e quebra numa CPU ou num sistema mais antigo que a tag declara (piso de glibc ou macOS mentindo,
  instrução fora do baseline).
- Wheel Linux com piso glibc 2.39 (o do runner ubuntu-24.04), que deixaria de fora o Colab (Ubuntu 22.04), o RHEL 8
  e clusters de universidade.
- Mudança de ABI depois da 1.0 que passe pelo CI.

## 3. Restrições

- O núcleo não usa `mdspan`, `<print>` nem `<generator>` (conferido em 2026-11-15). Então GCC 14 deve bastar, e o
  container manylinux_2_28 (gcc-toolset-14) é viável. Isso precisa ser confirmado em compilação real no 6a.
- O build não usa `-march=native`. O Highway despacha AVX2, AVX-512, NEON, SVE e SVE2 em runtime, e o baseline é o
  do compilador. Um binário por plataforma serve.
- No plano 5a, o build MSVC mediu 2,6x mais lento que o GCC (MinGW).
- O Python usa ctypes e não tem extensão CPython. Basta um wheel por plataforma, para todas as versões 3.11+.
- O loader Python (`python/src/volengine/_native.py`) procura a biblioteca em `_lib/` ou em `VOLENGINE_LIB` e confere
  `VE_ABI_VERSION`.
- A interface C exporta 133 funções `VE_API` hoje, sem structs públicas (configuração só por setters).
- O autor trabalha direto em master.

## 4. Abordagens consideradas

**Divisão.** Três subplanos, escritos e executados um de cada vez como no plano 5, nesta ordem:
- **6a, empacotamento e release** (§5.1);
- **6b, wrapper C++** (§5.2). Vem antes do congelamento porque é um segundo consumidor da interface C: como o Python
  foi, revela defeitos de ABI enquanto ainda dá para mudar;
- **6c, auditoria da ABI e 1.0** (§6).

**Compilador do wheel Windows (escolha do autor: 1).**
1. MinGW GCC com runtime estático. Mesma velocidade do build local; custa o MSYS2 no runner e um `.lib` gerado de
   `.def` para quem usa MSVC. **Escolhida.**
2. MSVC. Mais simples, mas o Windows fica 2,6x mais lento.
3. clang-cl. Velocidade nunca medida, e seria um terceiro toolchain.

**Como o wrapper chega ao usuário C++ (escolha do autor: A).**
- **A. Binário pronto, como o ExcelReader `cpp/`.** O `FetchContent` com `SOURCE_SUBDIR cpp` baixa da release da tag a
  biblioteca da plataforma. Quem consome compila só headers e recebe o binário MinGW rápido mesmo com MSVC. Depende
  de a release existir. **Escolhida.**
- B. Compilar do fonte. Funciona offline, mas o build leva minutos, exige o compilador C++23 completo, e o usuário MSVC
  fica 2,6x mais lento.
- C. As duas. Cobre tudo, ao custo de dois caminhos para testar.

**Contrato da ABI (escolha do Claude, aceita pelo autor).** Um arquivo versionado com cada declaração `VE_API` e cada
valor de enum ou `#define` de status, conferido contra `volengine.h` em todo CI, no papel do `PublicAPI.Shipped` dos
projetos .NET do autor. Mais abidiff no Linux, como a §5 da spec-mãe pede. Só o abidiff foi descartado: não roda em
Windows nem macOS, e sem informação de debug compara só nomes.

## 5. Desenho

### 5.1 Plano 6a: empacotamento e release

O modelo é o `release.yml` do FixedWidthParser (que o do ExcelReader repete): tag `v*` publica tudo, e uma tag com
`-` vira prerelease.

**Builds nativos**

| Plataforma | Runner | Onde compila | Toolchain |
|---|---|---|---|
| Linux glibc x64 | ubuntu-latest | container `manylinux_2_28_x86_64` | gcc-toolset-14 |
| Linux glibc arm64 | ubuntu-24.04-arm | container `manylinux_2_28_aarch64` | gcc-toolset-14 |
| Linux musl x64 | ubuntu-latest | container `musllinux_1_2_x86_64` | GCC do container |
| Linux musl arm64 | ubuntu-24.04-arm | container `musllinux_1_2_aarch64` | GCC do container |
| macOS arm64 | macos-latest | runner | Apple Clang, `MACOSX_DEPLOYMENT_TARGET=11.0` |
| macOS x64 | macos-15-intel | runner | Apple Clang, `MACOSX_DEPLOYMENT_TARGET=11.0` |
| Windows x64 | windows-latest | MSYS2 ucrt64 | GCC, `-static-libgcc -static-libstdc++` e winpthread estático |

- **Fallback do manylinux:** se o GCC 14 não compilar, usa-se `manylinux_2_34` (GCC 14 ou 15). O piso sobe para glibc
  2.34, que ainda cobre o Ubuntu 22.04. A decisão é registrada no plano.
- **Deployment target do macOS:** 11.0 é o mínimo que o arm64 aceita. Só sobe se o libc++ recusar algum recurso
  (erro de availability), e a razão é registrada.
- **Windows:** a DLL não pode depender de nenhuma DLL do MinGW. Um passo do CI confere isso com `objdump -p`, e a lista
  de imports só pode ter DLLs do sistema.
- **Runners arm64:** o ubuntu-24.04-arm é Neoverse com SVE2, então o caminho SVE do Highway passa a ser testado de
  verdade.

**Wheel**
- `python/pyproject.toml` continua com o hatchling.
- Entra um hook de build no padrão do `hatch_build.py` do FixedWidthParser. Ele força `py3-none-<tag>` e lê a tag do
  próprio binário: o piso de glibc pelos símbolos `GLIBC_2.N`, o piso de macOS pelo `LC_BUILD_VERSION` e o musl pelo
  `libc.musl-`.
- No Linux, o wheel passa pelo `auditwheel repair` (ou `auditwheel show` como checagem, se o binário já é
  autocontido).
- O piso declarado tem de ser o real. O §2 proíbe tag que mente.

**uv**
- `uv.lock` commitado em `python/`.
- As dependências de teste viram `[dependency-groups]` (PEP 735) no lugar do extra `test` (decisão do Claude).
- `ci.yml` troca `setup-python` mais `pip` por `astral-sh/setup-uv`, `uv sync` e `uv run pytest`, nas versões 3.14 e
  3.11, a mais nova e o piso.
- No release: `uv build --wheel`, a suíte pytest contra o wheel instalado em cada plataforma (incluindo os containers
  musl e manylinux), depois `uv publish` por Trusted Publishing.

**Release**
- Cada biblioteca vai para a GitHub Release como `volengine-native-<os>-<arch>.<ext>`:
  - `os` é `win`, `osx`, `linux` ou `linux-musl`;
  - `arch` é `x64` ou `arm64`;
  - no Windows, `.dll` mais `.lib` (gerado do `.def`, §6) e `.dll.a`.
- Vai também um SBOM SPDX.
- O nome do arquivo, o SONAME no Linux e o `install_name` no macOS (`@rpath/...`) têm de concordar com o que o loader
  procura depois que o `cpp/` baixa o asset (§5.2). Como fazer cabe ao builder; o ExcelReader usa `install_name_tool`.

**Versão**
- A tag é a fonte. O release injeta a versão no `pyproject.toml` e no `project(VERSION)` do CMake.
- No repositório, as duas ficam em `0.9.0.dev0`/`0.9.0` até a 1.0.

**CI fora do release**
- Um workflow `wheels.yml` monta os 7 wheels em todo push para master que toque `src/`, `include/`, `python/`,
  `CMakeLists.txt` ou o próprio workflow. Assim um release nunca compila uma plataforma pela primeira vez (como o
  `native-bindings.yml` do FixedWidthParser).

**Benchmark (o autor quer benchmark em todo plano)**
- No Ryzen 7 5700X, comparar o pipeline do TCC e o `quick.py` rodando sobre o wheel instalado contra o build local
  `gcc-bench`, na mesma sessão.
- Diferença acima do ruído é bug do 6a. Candidatos: runtime estático, flags ou baseline do Highway.
- No Windows, o wheel MinGW tem de bater com o build local MinGW.

**Ensaio**
- O 6a termina com a `v0.9.0` publicada de verdade no PyPI e na GitHub Release. A versão 0.x pode quebrar (§5).
- O autor confirmou em 2026-11-15 que o nome `volengine` está livre no PyPI e cadastrou o Trusted Publisher
  (repositório, `release.yml`).

### 5.2 Plano 6b: wrapper C++

O modelo é o `cpp/` do ExcelReader (`cpp/include/xl`, `FetchNativeLib.cmake`, `CopyNativeLibrary.cmake`, install e
config package). Os nomes abaixo são proposta do Claude; o builder pode ajustar.

**Layout**
- `cpp/include/volengine/volengine.hpp` traz o header-only, namespace `ve`, e inclui `<volengine/volengine.h>`.
- No repositório, o CMake do `cpp/` aponta para `../include`. Com `FetchContent` o repositório inteiro é clonado,
  então não precisa do symlink que o ExcelReader usa.

**CMake**
- `cpp/CMakeLists.txt` define `volengine::volengine`: um alvo INTERFACE com os headers, ligado ao alvo IMPORTED da
  biblioteca baixada.
- Também define:
  - `volengine_copy_native_library(alvo)`, obrigatório no Windows;
  - a variável de ambiente `VOLENGINE_NATIVE_LIB`, que aponta para um build local. O CI do PR usa isso. Com ela
    definida, nada é baixado;
  - a cache `VOLENGINE_VERSION`, que vem de `git describe --tags --exact-match` ou é passada explicitamente;
  - `VOLENGINE_INSTALL`, ligado só quando o `cpp/` é o projeto raiz. Gera install mais `volengineConfig.cmake`, e
    então `find_package(volengine 1.0)` funciona sem rede.
- Fora de uma tag e sem a variável de ambiente, o configure falha com uma mensagem clara.

**API do wrapper**
- RAII sobre cada handle (`ve_context`, specs, batches, fits, resultados).
- Erros por `std::expected<T, ve::Error>`, com a mensagem de `ve_last_error`.
- Entradas por `std::span<const double>`. Visões zero-copy como `std::span<const double>`, válidas enquanto o objeto
  dono existir.
- Sobrecargas de conveniência (uma spec ou uma lista, §7) só no wrapper, como a spec-mãe manda.
- A superfície segue a do Python: `Context`, `Spec`, `Fit` (critérios e diagnósticos como membros), lote, forecast,
  simulação, DCC e Copula-DCC.

**Testes**
- GoogleTest em `cpp/tests`, contra a biblioteca do build (`VOLENGINE_NATIVE_LIB`).
- Cada função `VE_API` é chamada pelo wrapper ao menos uma vez. O objetivo é exercitar a ABI inteira, e um teste lista
  as funções do contrato (§6) que o wrapper não usa.
- Alguns casos repetem fixtures de oráculo do Python para conferir que os números são os mesmos.
- Um projeto consumidor mínimo é configurado nos dois modos (`FetchContent` local e `find_package` depois de
  `cmake --install`) em Windows (MSVC sobre a DLL MinGW), Linux e macOS.

**Benchmark**
- Google Benchmark do custo do wrapper contra a chamada C direta num ajuste e num lote. Esperado: ruído.

### 6. Plano 6c: auditoria da ABI e 1.0

**Auditoria**
- Uma passada nas 133 funções (mais as que o 6b acrescentar) contra as regras da §5 da spec-mãe.
- Ela procura o que exigiria **mudar ou remover** algo depois da 1.0: nomes, ordem de parâmetros, tipos, ownership,
  valores de enum e semântica de status.
- Acrescentar algo depois da 1.0 não quebra a ABI, então lacunas que só pedem função nova não bloqueiam.
- Cada achado é corrigido antes da 1.0, ou registrado como aceito com o motivo.

**Contrato**
- Um arquivo versionado (nome e formato ficam com o builder; por exemplo `include/volengine/volengine.abi`) com cada
  declaração `VE_API` normalizada e cada valor de enum ou `#define`.
- Um script gera o arquivo a partir do `volengine.h`. O CI falha se o header e o arquivo divergirem: a mudança tem de
  ser explícita no diff.
- Depois da 1.0, o script também falha se uma linha do contrato publicado sumir ou mudar. Só acréscimo passa.
- O `.def` do Windows (§5.1) sai do mesmo arquivo, que vira a fonte única dos símbolos exportados.

**abidiff**
- O release anexa o `abidw` da `.so` Linux x64 (build com `-g`).
- Um job do CI compila a `.so` atual e roda `abidiff` contra o XML da última release.
- Da `v0.9.0` até a 1.0 o job só informa. A partir da `v1.0.0`, mudança incompatível falha o CI.

**Versão**
- `VE_ABI_VERSION` passa a 1 na 1.0, e o loader Python e o wrapper conferem a igualdade.
- Pacote e projeto ficam em 1.0.0. O major do semver acompanha o `VE_ABI_VERSION` (decisão do Claude).

**Documentação**
- O README ganha `pip install`, um exemplo Python (ajuste, forecast, DCC) e um exemplo C++ (`FetchContent`).
- `docs/guide/` ganha um guia rápido por linguagem.
- O CHANGELOG ganha a entrada 1.0, e o CITATION.cff recebe a versão.

**Release**
- `v1.0.0-rc.1` pelo pipeline inteiro, e depois `v1.0.0`.

## 7. Deixado ao builder

- Nomes de arquivos, scripts, jobs e do formato do contrato da ABI.
- Como fazer o nome do asset, o SONAME e o install_name concordarem.
- `auditwheel repair` ou só `auditwheel show`.
- Detalhes do hook de hatchling (copiar do FixedWidthParser e adaptar).
- Organização interna do `volengine.hpp` (um header ou vários sob `cpp/include/volengine/`).
- Quais fixtures de oráculo o teste do wrapper repete.

## 8. Decisões do Claude (para conferir)

- Sem sdist.
- `manylinux_2_28` como alvo, com fallback `manylinux_2_34`.
- `MACOSX_DEPLOYMENT_TARGET=11.0` nas duas arquiteturas.
- `[dependency-groups]` no lugar do extra `test`.
- Python 3.14 e 3.11 no CI.
- `wheels.yml` em todo push para master que toque o código.
- `v0.9.0` publicada de verdade no PyPI como ensaio.
- `VE_ABI_VERSION = 1` e semver major 1 na 1.0.
- O `.def` gerado do contrato da ABI.
- Nomes `ve::`, `volengine::volengine`, `VOLENGINE_NATIVE_LIB`, `VOLENGINE_VERSION`, `VOLENGINE_INSTALL`.
