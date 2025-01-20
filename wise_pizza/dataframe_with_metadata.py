import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataFrameWithMetadata(pd.DataFrame):
    def __init__(
        self,
        *args,
        name: str = None,
        description: str = None,
        column_descriptions=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.attrs["name"] = name or ""  # Store DataFrame name
        self.attrs["description"] = description or ""  # Store DataFrame description
        self.attrs["column_descriptions"] = {}

        if column_descriptions:
            column_descriptions = {
                k: v for k, v in column_descriptions.items() if k in self.columns
            }
            if column_descriptions:
                self.attrs["column_descriptions"] = column_descriptions
            else:
                logger.warning(
                    "None of the column descriptions provided matched the DataFrame columns"
                )

    def to_markdown(self, index: bool = True, **kwargs):
        # Start with DataFrame description if it exists
        output = []
        if self.attrs["name"]:
            output.append(f"Table name: {self.attrs['name']}\n")

        if self.attrs["description"]:
            output.append(f"Table description: {self.attrs['description']}\n")

        if not self.attrs["column_descriptions"]:
            output.append(super().to_markdown(index=index, **kwargs))
            return "\n".join(output)

        desc_row = " | ".join(
            (["---"] if index else [])
            + [self.attrs["column_descriptions"].get(col, "") for col in self.columns]
        )
        original_md = super().to_markdown(index=index, **kwargs)
        header_end = original_md.index("\n|")
        output.append(
            original_md[:header_end] + "\n|" + desc_row + original_md[header_end:]
        )
        return "\n".join(output)

    def head(self, n: int = 5):
        out = DataFrameWithMetadata(super().head(n))
        out.attrs = self.attrs
        return out


if __name__ == "__main__":
    # Usage example:
    df = DataFrameWithMetadata(
        {"a": [1, 2], "b": [3, 4]},
        description="Description for the DataFrame",
        name="DataFrame Name",
        column_descriptions={
            "a": "Description for column a",
            "b": "Description for column b",
        },
    )

    md = df.to_markdown()
    print(md)
    md2 = df.to_markdown(index=False)
    print(md2)
    print("yay!")
    # This would raise an error:
    # df = DescribedDataFrame({'a': [1]}, descriptions={'nonexistent': 'Description'})
