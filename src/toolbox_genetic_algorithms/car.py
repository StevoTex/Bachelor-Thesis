from evolution.interfaces import Allele, Individual
from utilities import consumption_model as cm


class FinalDrive(Allele):
    _min = 1
    _max = 6


class RollRadius(Allele):
    _min = 0.2
    _max = 0.5


class Gear(Allele):
    _min = 0.5
    _max = 2


class Car(Individual):
    _Blueprint = {
        'genotype': [FinalDrive, RollRadius, Gear, Gear, Gear],
        'genotype_labels': ['Final Drive', 'Roll Radius', 'Gear 3', 'Gear 4', 'Gear 5'],
        'goals': [-4, -2, -1, -1],  # consumption, elasticity 3, elasticity 4, elasticity 5
        'phenotype_labels': ['Consumption', 'Elasticity 3', 'Elasticity 4', 'Elasticity 5'],
    }

    def _calculate_phenotype(self):
        # extract the alleles from the genotype
        final_drive_ratio, roll_radius, gear3, gear4, gear5 = self.get_genotype()

        # run the simulation
        consumption, ela_3, ela_4, ela_5 = cm.run_simulation(final_drive_ratio, roll_radius, gear3, gear4, gear5)

        # return the phenotype
        return consumption, ela_3, ela_4, ela_5

    def _enforce_constraints(self):
        # putting the gears in the correct order
        final_drive_ratio, roll_radius, gear3, gear4, gear5 = self.get_genotype()
        gear3, gear4, gear5 = sorted([gear3, gear4, gear5], reverse=True)

        # make sure, the simulation model is not broken because of edge cases
        g3 = max(0.1, gear3.get(normalized=True))
        gear3.set(g3, normalized=True)
        g4 = max(0.05, min(0.95, gear4.get(normalized=True)))
        gear4.set(g4, normalized=True)
        g5 = min(0.9, gear5.get(normalized=True))
        gear5.set(g5, normalized=True)
        # TODO: Unfortunately, this does not solve the problem ... here is still some work to do

        # return the repaired genotype
        self._genotype = final_drive_ratio, roll_radius, gear3, gear4, gear5
