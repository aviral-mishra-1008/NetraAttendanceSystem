# Generated by Django 5.0.1 on 2024-05-23 18:21

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Netra', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='mnnit',
            old_name='sections',
            new_name='section',
        ),
        migrations.DeleteModel(
            name='Professor',
        ),
    ]
