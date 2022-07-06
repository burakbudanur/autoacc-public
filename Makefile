all:
	make figmodels
	make figfits
	make figlockdowns
	make figfitscatter
	make figbootstrap
	make figparscatter

figmodels:
	cd python/models; python models.py

figfits:
	cd python/fits; python fits.py

figlockdowns:
	cd python/lockdowns; python lockdowns.py

figfitscatter:
	cd python/fitscatter; python fitscatter.py

figbootstrap:
	cd python/bootstrap; python plot_bootstrap.py

figparscatter:
	cd python/bootstrap; python parameter_scatter.py
