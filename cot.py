import re
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import generate, prompt_template
from inspect_ai.scorer import Score, CORRECT, INCORRECT, scorer, accuracy, stderr
from inspect_ai.model import get_model

def extract_content_between_tags(xml_string, tag):
    pattern = fr'<{tag}>(.*?)</{tag}>'
    return re.findall(pattern, xml_string)


PROMPT_TEMPLATE: str = "Think step by step {prompt}. Show all your steps one by one, and then give the final answer"
samples = [
    Sample(input="Thirteen years before humans landed on the Moon in 1969, the Soviet Union invaded one of its client states to suppress a revolution. This client state was once ruled by a Middle Eastern empire starting with the letter 'O' in the 16th century. What was the first name of the last king of that empire.", target="Mehmed"),
    Sample(input=''' In an enchanted library, each book has a unique title encoded in a specific way. The title of one book is "L3t5". The encoding rule is as follows: Each letter in the title is replaced by its ASCII value. The numbers in the title are kept as they are. The sum of all the resulting numbers is calculated. If the sum is even, the book is in the "Mystic" section; if odd, it is in the "Arcane" section. Another book has the title "H4x0r". What section is it in? ''',
           target="Mystic"),

    Sample(input='''
           The year Alexander Graham Bell patented the telephone, Queen Victoria was proclaimed Empress of India. Twenty-seven years later, when the Wright brothers made their first powered flight, a U.S. president was in office. What's the product of the number of letters in this president's first name and the years between these two events?
           ''', target="216"),
    Sample(input='''
           When Napoleon III was captured by Prussian forces in the Battle of Sedan, a famous American author who wrote about a white whale was still alive. The year this author died, Alexander Graham Bell had already made the first phone call saying 'Mr. Watson, come here.' The number of letters in this author's first name, multiplied by the years between Bell's call and the author's death, equals what number?''', target="90"),
    Sample(
        input='''
        The year Queen Victoria died, Guglielmo Marconi sent the first transatlantic radio signal. This inventor later shared the Nobel Prize in Physics the same year Robert Peary claimed to reach the North Pole. The number of letters in this inventor's last name, multiplied by the years between his radio signal and his Nobel Prize, equals what number?
        ''',
        target="56"
    )
]

dataset = MemoryDataset(samples, name="multihop_dataset")

dummy_solver = [prompt_template(PROMPT_TEMPLATE), generate()]

@scorer(metrics=[accuracy(), stderr()])
def model_based_scorer():
    async def score(state, target):
        prompt = f"""
        Is the following output correct? Just focus if the final answer is correct, and not any unnecesary details

        Question: {state.input}
        Output: {state.output.completion}
        Correct answer: {target.text}

        Respond with "Yes" or "No" in <answer></answer> tags and an explanation in <explanation></explanation> tags
        """
        response = await get_model("openai/gpt-4o-mini").generate(prompt)

        if extract_content_between_tags(response.completion.strip(), "answer")[0].lower() == "yes":
            return Score(value=CORRECT, explanation=response.completion)
        else:
            return Score(
                value=INCORRECT,
                explanation=response.completion,
            )

    return score

@task
def dummy_task():
    return Task(
        dataset=dataset,  # Use the dummy dataset
        solver=dummy_solver,  # Use the dummy solver
        scorer=model_based_scorer(),  # Use the model-based scorer
    )

