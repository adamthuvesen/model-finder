RUN := uv run --locked --extra models --extra deep

# oauthlib and thrift are held back by databricks-sql-connector, which only the optional
# `databricks` extra installs. Remove these when the connector allows the fixed versions.
REVIEWED_VULNS := --ignore-vuln CVE-2026-49272 --ignore-vuln PYSEC-2026-4114 --ignore-vuln PYSEC-2026-3925 --ignore-vuln PYSEC-2026-3926 --ignore-vuln PYSEC-2026-3927

.PHONY: check audit lock lint types boundaries rules deps dead test workflows secrets vulnerabilities

# Offline gate. CI runs the same targets.
check: lock lint types boundaries rules deps dead test

# Needs the network for the vulnerability database.
audit: workflows secrets vulnerabilities

lock:
	uv lock --check

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

types:
	$(RUN) basedpyright --level error

boundaries:
	$(RUN) lint-imports

rules:
	$(RUN) ast-grep test --skip-snapshot-tests
	$(RUN) ast-grep scan --error

deps:
	$(RUN) deptry .

dead:
	$(RUN) vulture

test:
	$(RUN) pytest --quiet --cov --cov-fail-under=89

workflows:
	$(RUN) zizmor --offline .github/workflows
	$(RUN) actionlint

secrets:
	gitleaks git --no-banner --redact

vulnerabilities:
	uv export --quiet --frozen --all-extras --all-groups --no-hashes --no-emit-project --output-file .audit-requirements.txt
	$(RUN) pip-audit --requirement .audit-requirements.txt --no-deps --disable-pip --progress-spinner off $(REVIEWED_VULNS); status=$$?; rm -f .audit-requirements.txt; exit $$status
