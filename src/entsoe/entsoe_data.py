import pandas as pd

class EntsoeData:
    def __init__(
            self, 
            CLIENT,
            country='FR', 
            start="2025-10-11", 
            end="2025-10-12",
    ):
        # Initialize the primary inputs
        self.CLIENT = CLIENT
        self.country = country
        self.start = pd.Timestamp(start, tz="Europe/Paris")
        self.end = pd.Timestamp(end, tz="Europe/Paris")
    
    # Get DA price
    def getPriceDA(self):
        df = pd.DataFrame(self.CLIENT.query_day_ahead_prices(
            country_code=self.country,
            start=self.start,
            end=self.end,
        ))
        df.columns = ['day-ahead']
        return df
    
    # Get imbalance price
    def getPriceIM(self):
        df = pd.DataFrame(self.CLIENT.query_imbalance_prices(
            country_code=self.country,
            start=self.start,
            end=self.end,
            psr_type=None,
        ))
        return df

    # Get FCR contracted price
    def getPriceFCR_contracted(self):
        df = pd.DataFrame(self.CLIENT.query_contracted_reserve_prices(
            country_code=self.country,
            process_type = "A52",
            type_marketagreement_type = "A01",
            start=self.start,
            end=self.end,
            psr_type=None,
        ))
        return df["Symmetric"]

    # Get FCR contracted volume
    def getVolumeFCR_contracted(self):
        df = pd.DataFrame(self.CLIENT.query_contracted_reserve_amount(
            country_code=self.country,
            process_type = "A52",
            type_marketagreement_type = "A01",
            start=self.start,
            end=self.end,
            psr_type=None,
        ))
        return df["Symmetric"]

    # Get FCR activated volume
    def getVolumeFCR_activation(self):
        df = pd.DataFrame(self.CLIENT.query_activated_balancing_energy(
            country_code=self.country,
            start=self.start,
            end=self.end,
            business_type = "A95",
            psr_type=None,
        ))
        df.columns = df.columns.droplevel()
        return df

    # Get FCR activated price
    def getPriceFCR_activation(self):
        df = pd.DataFrame(self.CLIENT.query_activated_balancing_energy_prices(
            country_code=self.country,
            start=self.start,
            end=self.end,
            process_type='A16',
            psr_type=None,
            business_type = "A95",
            standard_market_product=None, 
            original_market_product=None,
        ))
        return df
