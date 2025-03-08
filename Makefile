pytest:
	@sh bash/execute_pytest.sh $(path)

ingestion:
	uv run ingestion.py

linters:
	@sh bash/execute_linters.sh $(path)

app:
	streamlit run app.py