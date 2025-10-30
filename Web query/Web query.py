import typer

app = typer.Typer()

@app.command()
def query(task_id: int):
    # Replace this with actual APL task lookup logic
    if task_id == 99:
        typer.echo("Task 99: [Task's description or details here]")
    else:
        typer.echo(f"No information found for task {task_id}")

if __name__ == "__main__":
    app()
