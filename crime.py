from sklearn import neighbors


class Crime:

    def __init__(self, id, report_datetime, incident_datetime, desc, neighborhood, address):
        self.id = id
        self.report_datetime = report_datetime
        self.incident_datetime = incident_datetime
        self.desc = desc
        self.neighborhood = neighborhood
        self.address = address

    @staticmethod
    def from_csv_entry(string):
        values = string.split(",")
        assert(len(values) == 6)
        return Crime(values[0], values[1], values[2], values[3], values[4], values[5])
    
    